# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/builder.py and https://github.com/NVlabs/Sana.
# This implementation is a modified version for class-to-image generation.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from ....apps.utils.init import init_modules
from ...models.ops.input_embed import AdaptivePatchEmbed
from ...models.ops.label_embed import LabelEmbedder
from ...models.ops.timestep_embed import timestep_embedding
from .base import BaseDiffusionModel, BaseDiffusionModelConfig

__all__ = ["SanaClsConfig", "SanaCls", "dc_ae_sana_cls_xl_in_512px"]


@dataclass
class SanaClsConfig(BaseDiffusionModelConfig):
    name: str = "SanaCls"

    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    mlp_ratio: float = 2.5
    post_norm: bool = False
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    unconditional: bool = False
    patch_kernel_size: Optional[int] = None

    num_inference_steps: int = 250
    train_scheduler: str = "SanaScheduler"
    eval_scheduler: str = "SanaScheduler"

    ffn_mode: str = "GLUMBConv"
    use_linear_attn: bool = False
    attention_head_dim: int = 32


def modulate(x, shift, scale, base: float = 1):
    return x * (base + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
#                                 Core SanaCls Model                                #
#################################################################################


class FinalLayer(nn.Module):
    """
    The final layer of SanaCls.
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


class SanaCls(BaseDiffusionModel):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(self, cfg: SanaClsConfig):
        super().__init__(cfg)
        self.cfg: SanaClsConfig

    def build_model(self):
        from .sana_utils.sana_block import SanaClsTransformerBlock

        self.out_channels = (
            self.cfg.in_channels if self.cfg.train_scheduler != "GaussianDiffusion" else self.cfg.in_channels * 2
        )
        self.x_embedder = AdaptivePatchEmbed(
            self.cfg.input_size,
            self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.hidden_size,
            bias=True,
            share_weights=self.cfg.adaptive_channel_share_weights,
            kernel_size=self.cfg.patch_kernel_size,
        )
        self.t_embedder = TimestepEmbedder(self.cfg.hidden_size)
        if not self.cfg.unconditional:
            self.y_embedder = LabelEmbedder(self.cfg.num_classes, self.cfg.hidden_size, self.cfg.class_dropout_prob)
        self.blocks = nn.ModuleList(
            [
                SanaClsTransformerBlock(
                    self.cfg.hidden_size,
                    self.cfg.mlp_ratio,
                    ffn_mode=self.cfg.ffn_mode,
                    use_linear_attn=self.cfg.use_linear_attn,
                    attention_head_dim=self.cfg.attention_head_dim,
                )
                for _ in range(self.cfg.depth)
            ]
        )
        self.final_layer = FinalLayer(self.cfg.hidden_size, self.cfg.patch_size, self.out_channels)

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["x_embedder", "t_embedder", "y_embedder", "blocks", "final_layer"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)
        for name, _ in self.named_parameters(recurse=False):
            raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "dc-ae":
            if "ema" in checkpoint:
                checkpoint = next(iter(checkpoint["ema"].values()))
            else:
                checkpoint = checkpoint["model_state_dict"]
            self.get_trainable_modules_list().load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def initialize_weights(self):
        # apply the standard truncated normal initialization to all modules as the default initialization
        init_modules(self, init_type="trunc_normal@0.02")

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

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def enable_activation_checkpointing(self, mode: str):
        for i in range(len(self.blocks)):
            self.blocks[i] = checkpoint_wrapper(self.blocks[i], preserve_rng_state=False)

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Forward pass of SanaCls.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        info = {}
        if self.cfg.count_nfe:
            self.nfe += 1
        x = self.x_embedder(x)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        if self.cfg.unconditional:
            c = t
        else:
            y = self.y_embedder(y, self.training)  # (N, D)
            c = t + y  # (N, D)

        H, W = self.cfg.input_size // self.cfg.patch_size, self.cfg.input_size // self.cfg.patch_size
        for block in self.blocks:
            x = block(x, c, H, W)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x, info


def dc_ae_sana_cls_xl_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=sana_cls sana_cls.depth=28 sana_cls.hidden_size=1152 sana_cls.attention_head_dim=32 sana_cls.in_channels={in_channels} sana_cls.patch_size=1 "
        "sana_cls.train_scheduler=SiTSampler sana_cls.eval_scheduler=ODE_dopri5 "
        f"sana_cls.pretrained_path={'null' if pretrained_path is None else pretrained_path} "
        "fid.ref_path=assets/data/fid/imagenet_train_512.npz"
    )
