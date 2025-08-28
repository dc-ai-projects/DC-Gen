# DiT was introduced by William Peebles and Saining Xie in "Scalable Diffusion Models with Transformers", see http://arxiv.org/abs/2212.09748.
# The original implementation is by Meta Platforms, Inc. and affiliates, licensed under CC-BY-NC. See https://github.com/facebookresearch/DiT.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn import functional as F

from ....apps.utils.init import init_modules
from ....models.nn.norm import build_norm
from ....models.nn.ops import GLUMBConv, ResidualBlock
from ....models.utils.network import get_submodule_weights
from ...models.ops.input_embed import AdaptivePatchEmbed
from ...models.ops.label_embed import LabelEmbedder
from ...models.ops.timestep_embed import timestep_embedding
from .base import BaseDiffusionModel, BaseDiffusionModelConfig

__all__ = ["DiTConfig", "DiT", "dc_ae_dit_xl_in_512px"]


@dataclass
class DiTConfig(BaseDiffusionModelConfig):
    name: str = "DiT"

    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    post_norm: bool = False
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True
    unconditional: bool = False
    patch_kernel_size: Optional[int] = None

    eval_scheduler: str = "GaussianDiffusion"
    num_inference_steps: int = 250
    train_scheduler: str = "GaussianDiffusion"

    freeze_backbone: bool = False
    head_only: bool = False
    patch_ffn_depth: int = 0


def modulate(x, shift, scale, base: float = 1):
    return x * (base + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, post_norm=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.post_norm = post_norm
        if not post_norm:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        else:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 4 * hidden_size, bias=True))

    def forward(self, x, c):
        if not self.post_norm:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
            x = x + modulate(self.norm1(self.attn(x)), shift_msa, scale_msa, base=0)
            x = x + modulate(self.norm2(self.mlp(x)), shift_mlp, scale_mlp, base=0)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class AdaptiveFinalLayer(nn.Module):
    """
    The final layer of DiT with adaptive channels.
    """

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, learn_sigma: bool, share_weights: bool):
        super().__init__()
        self.patch_size = patch_size
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.learn_sigma = learn_sigma
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear_out_channels = patch_size * patch_size * out_channels
        if share_weights:
            self.linear = nn.Linear(hidden_size, self.linear_out_channels, bias=True)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, c: torch.Tensor, out_channels: int) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        linear_out_channels = self.patch_size * self.patch_size * out_channels
        if self.learn_sigma:
            x = F.linear(
                x,
                torch.cat(
                    [
                        self.linear.weight[:linear_out_channels],
                        self.linear.weight[
                            self.linear_out_channels // 2 : self.linear_out_channels // 2 + linear_out_channels
                        ],
                    ]
                ),
                torch.cat(
                    [
                        self.linear.bias[:linear_out_channels],
                        self.linear.bias[
                            self.linear_out_channels // 2 : self.linear_out_channels // 2 + linear_out_channels
                        ],
                    ]
                ),
            )
        else:
            x = F.linear(x, self.linear.weight[:linear_out_channels], self.linear.bias[:linear_out_channels])
        return x


class DiT(BaseDiffusionModel):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__(cfg)
        self.cfg: DiTConfig

        if cfg.freeze_backbone or cfg.head_only:
            # freeze all parameters
            for parameter in self.parameters():
                parameter.requires_grad = False
            # unfreeze the parameters of selected modules
            if cfg.head_only:
                unfreeze_modules = [self.final_layer.linear]
            else:
                unfreeze_modules = [self.x_embedder, self.patch_ffn, self.final_layer]
            for m in unfreeze_modules:
                if m is not None:
                    for parameter in m.parameters():
                        parameter.requires_grad = True

    def build_model(self):
        self.out_channels = self.cfg.in_channels * 2 if self.cfg.learn_sigma else self.cfg.in_channels

        self.x_embedder = AdaptivePatchEmbed(
            self.cfg.input_size,
            self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.hidden_size,
            bias=True,
            share_weights=self.cfg.adaptive_channel_share_weights,
            kernel_size=self.cfg.patch_kernel_size,
        )

        if self.cfg.patch_ffn_depth > 0:
            self.patch_ffn = nn.Sequential(
                *[
                    ResidualBlock(
                        main=GLUMBConv(
                            self.cfg.hidden_size,
                            self.cfg.hidden_size,
                            expand_ratio=self.cfg.mlp_ratio,
                            use_bias=True,
                            norm=None,
                            act_func=("silu", "silu", None),
                        ),
                        shortcut=nn.Identity(),
                        pre_norm=build_norm("trms2d", self.cfg.hidden_size),
                    )
                    for _ in range(self.cfg.patch_ffn_depth)
                ]
            )
        else:
            self.patch_ffn = None
        self.t_embedder = TimestepEmbedder(self.cfg.hidden_size)
        if not self.cfg.unconditional:
            self.y_embedder = LabelEmbedder(self.cfg.num_classes, self.cfg.hidden_size, self.cfg.class_dropout_prob)
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, self.cfg.hidden_size), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.cfg.hidden_size, self.cfg.num_heads, mlp_ratio=self.cfg.mlp_ratio, post_norm=self.cfg.post_norm
                )
                for _ in range(self.cfg.depth)
            ]
        )
        if self.cfg.adaptive_channel:
            self.final_layer = AdaptiveFinalLayer(
                self.cfg.hidden_size,
                self.cfg.patch_size,
                self.out_channels,
                self.cfg.learn_sigma,
                share_weights=self.cfg.adaptive_channel_share_weights,
            )
        else:
            self.final_layer = FinalLayer(self.cfg.hidden_size, self.cfg.patch_size, self.out_channels)

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["x_embedder", "patch_ffn", "t_embedder", "y_embedder", "blocks", "final_layer"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)
        for name, _ in self.named_parameters(recurse=False):
            if name in ["pos_embed"]:
                pass
            else:
                raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source in ["dit", "dc-adapt"]:
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            self.load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-ae":
            if "ema" in checkpoint:
                checkpoint = next(iter(checkpoint["ema"].values()))
            else:
                checkpoint = checkpoint["model_state_dict"]
            self.get_trainable_modules_list().load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-ae-1.0":
            checkpoint = next(iter(checkpoint["ema"].values()))
            self.load_state_dict(get_submodule_weights(checkpoint, "dit."))
        elif self.cfg.pretrained_source == "dc-ae-fsdp":
            checkpoint = next(iter(checkpoint["ema"].values()))
            self.load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.weight.initialized = True
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    module.bias.initialized = True

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.initialized = True

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.cfg.patch_kernel_size is not None:
            init_modules(self.x_embedder, init_type="trunc_normal@0.02")
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)
            self.x_embedder.proj.weight.initialized = True
            self.x_embedder.proj.bias.initialized = True

        # Initialize label embedding table:
        if not self.cfg.unconditional:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
            self.y_embedder.embedding_table.weight.initialized = True

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        self.t_embedder.mlp[0].weight.initialized = True
        self.t_embedder.mlp[2].weight.initialized = True

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            block.adaLN_modulation[-1].weight.initialized = True
            block.adaLN_modulation[-1].bias.initialized = True

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        self.final_layer.adaLN_modulation[-1].weight.initialized = True
        self.final_layer.adaLN_modulation[-1].bias.initialized = True
        self.final_layer.linear.weight.initialized = True
        self.final_layer.linear.bias.initialized = True

        # Initialize patch_ffn
        if self.patch_ffn is not None:
            init_modules(self.patch_ffn, init_type="trunc_normal@0.02")
            # zero out the output of the patch_ffn
            for block in self.patch_ffn:
                assert isinstance(block, ResidualBlock) and isinstance(block.main, GLUMBConv)
                nn.init.constant_(block.main.point_conv.weight, 0)
                nn.init.constant_(block.main.point_conv.bias, 0)
                block.main.point_conv.weight.initialized = True
                block.main.point_conv.bias.initialized = True

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        p = self.x_embedder.patch_size[0]
        c = x.shape[2] // p**2
        assert p**2 * c == x.shape[2]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def enable_activation_checkpointing(self, mode: str):
        for i in range(len(self.blocks)):
            self.blocks[i] = checkpoint_wrapper(self.blocks[i], preserve_rng_state=False)

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        info = {}
        if self.cfg.count_nfe:
            self.nfe += 1
        in_channels = x.shape[1]

        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        if self.patch_ffn is not None:
            H = W = int(x.shape[1] ** 0.5)
            x = x.unflatten(1, (H, W)).permute(0, 3, 1, 2)
            x = self.patch_ffn(x)
            x = x.flatten(2, 3).permute(0, 2, 1)
        x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        t_emb = self.t_embedder(t)  # (N, D)
        if self.cfg.unconditional:
            c = t_emb
        else:
            y_emb = self.y_embedder(y, self.training)  # (N, D)
            c = t_emb + y_emb  # (N, D)
        for i, block in enumerate(self.blocks):
            x = block(x, c)  # (N, T, D)
        if self.cfg.head_only:
            x = x.detach()
            c = c.detach()
        if self.cfg.adaptive_channel:
            x = self.final_layer(x, c, out_channels=in_channels)
        else:
            x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        if self.cfg.learn_sigma and not self.training:
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                pass
            elif self.cfg.eval_scheduler in ["UniPC", "DPMSolverSinglestep", "ODE_dopri5", "ODE_heun2", "UniSampler"]:
                x = x[:, : x.shape[1] // 2]
            else:
                raise ValueError(f"eval scheduler {self.cfg.eval_scheduler} is not supported with learn_sigma=True")

        return x, info


def dc_ae_dit_xl_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=dit dit.depth=28 dit.hidden_size=1152 dit.num_heads=16 dit.in_channels={in_channels} dit.patch_size=1 "
        f"dit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_sit_xl_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=dit dit.depth=28 dit.hidden_size=1152 dit.num_heads=16 dit.in_channels={in_channels} dit.patch_size=1 dit.learn_sigma=False dit.train_scheduler=SiTSampler dit.eval_scheduler=ODE_dopri5 "
        f"dit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )
