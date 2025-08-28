# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana_multi_scale.py.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from ....models.utils.network import get_device
from .base import BaseT2IDiffusionModel, BaseT2IDiffusionModelConfig

__all__ = ["SanaT2IConfig", "SanaT2I"]


@dataclass
class SanaT2IConfig(BaseT2IDiffusionModelConfig):
    name: str = "SanaT2I"
    eval_scheduler: str = "DPMS"
    train_scheduler: str = "SanaScheduler"
    num_inference_steps: int = 20

    patch_size: int = 1
    hidden_size: int = 2240
    depth: int = 20
    pos_embed_type: str = "sincos"

    # caption embedder
    caption_channels: int = 2304
    class_dropout_prob: float = 0.1
    text_max_length: int = 300
    y_norm_scale_factor: float = 0.01
    norm_eps: float = 1e-5

    text_encoder_name: str = "google/gemma-2-2b-it"

    # SanaBlocks
    num_heads: int = 20
    mlp_ratio: float = 2.5
    mlp_acts: tuple[Optional[str]] = ("silu", "silu", None)
    drop_path: float = 0.0
    qk_norm: bool = False
    linear_head_dim: int = 32
    cross_norm: bool = False


class T2IAdaptiveFinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.patch_size = patch_size
        self.out_channels = out_channels

    def forward(self, x, t, out_channels):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        linear_out_channels = self.patch_size * self.patch_size * out_channels
        x = F.linear(x, self.linear.weight[:linear_out_channels], self.linear.bias[:linear_out_channels])
        return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class SanaMSBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        qk_norm=False,
        mlp_acts=("silu", "silu", None),
        linear_head_dim=32,
        cross_norm=False,
    ):
        super().__init__()
        from timm.layers import DropPath

        from .sana_blocks.attention import LiteLA, MultiHeadCrossAttention
        from .sana_blocks.glumb import SanaGLUMBConv

        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self_num_heads = hidden_size // linear_head_dim
        self.attn = LiteLA(hidden_size, hidden_size, heads=self_num_heads, eps=1e-8, qk_norm=qk_norm)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, qk_norm=cross_norm)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SanaGLUMBConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            expand_ratio=mlp_ratio,
            use_bias=(True, True, False),
            norm=(None, None, None),
            act_func=mlp_acts,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, mask=None, HW=None):
        B = x.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)

        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        x = x + self.cross_attn(x, y, mask)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), HW=HW))
        return x


class SanaT2I(BaseT2IDiffusionModel):
    def __init__(self, cfg: SanaT2IConfig):
        super().__init__(cfg)
        self.cfg: SanaT2IConfig

    def build_model(self):
        from ...models.ops.caption_embed import CaptionEmbedder
        from ...models.ops.input_embed import PatchEmbedMS
        from ...models.ops.timestep_embed import TimestepEmbedder
        from .sana_blocks.norm import SanaRMSNorm

        self.patch_nums = self.cfg.input_size // self.cfg.patch_size
        self.patch_size = self.cfg.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_nums**2, self.cfg.hidden_size), requires_grad=False)

        self.x_embedder = PatchEmbedMS(
            patch_size=self.cfg.patch_size,
            in_channels=self.cfg.in_channels,
            embed_dim=self.cfg.hidden_size,
            bias=True,
        )

        self.t_embedder = TimestepEmbedder(hidden_size=self.cfg.hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(self.cfg.hidden_size, 6 * self.cfg.hidden_size, bias=True))

        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(
            in_channels=self.cfg.caption_channels,
            hidden_size=self.cfg.hidden_size,
            uncond_prob=self.cfg.class_dropout_prob,
            act_layer=approx_gelu,
            token_num=self.cfg.text_max_length,
            text_encoder_name=self.cfg.text_encoder_name,
        )

        self.attention_y_norm = SanaRMSNorm(
            dim=self.cfg.hidden_size, scale_factor=self.cfg.y_norm_scale_factor, eps=self.cfg.norm_eps
        )

        drop_path = [x.item() for x in torch.linspace(0, self.cfg.drop_path, self.cfg.depth)]

        self.blocks = nn.ModuleList(
            [
                SanaMSBlock(
                    hidden_size=self.cfg.hidden_size,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    drop_path=drop_path[i],
                    qk_norm=self.cfg.qk_norm,
                    mlp_acts=self.cfg.mlp_acts,
                    linear_head_dim=self.cfg.linear_head_dim,
                    cross_norm=self.cfg.cross_norm,
                )
                for i in range(self.cfg.depth)
            ]
        )

        self.final_layer = T2IAdaptiveFinalLayer(
            hidden_size=self.cfg.hidden_size, patch_size=self.cfg.patch_size, out_channels=self.cfg.in_channels
        )

    def initialize_weights(self):
        from .sana_blocks.pos_embed import RopePosEmbed, get_2d_sincos_pos_embed

        def _basic_init(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                module.weight.initialized = True
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    module.bias.initialized = True

        self.apply(_basic_init)

        if self.cfg.pos_embed_type == "sincos":
            pos_embed = (
                torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_nums))
                .unsqueeze(0)
                .to(torch.float32)
            )
        elif self.cfg.pos_embed_type == "3d_rope":
            pos_embed = RopePosEmbed(theta=10000, axes_dim=[0, 16, 16])
        else:
            raise ValueError(f"Unknown pos_embed_type: {self.pos_embed_type}")
        self.pos_embed.data.copy_(pos_embed)
        self.pos_embed.initialized = True

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.x_embedder.proj.weight.initialized = True
        self.x_embedder.proj.bias.initialized = True

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        self.t_embedder.mlp[0].weight.initialized = True
        self.t_embedder.mlp[2].weight.initialized = True
        self.t_block[1].weight.initialized = True

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        self.y_embedder.y_proj.fc1.weight.initialized = True
        self.y_embedder.y_proj.fc2.weight.initialized = True

        nn.init.ones_(self.attention_y_norm.weight)
        self.attention_y_norm.weight.data.mul_(self.cfg.y_norm_scale_factor)
        self.attention_y_norm.weight.initialized = True

        for block in self.blocks:
            nn.init.normal_(block.scale_shift_table, mean=0.0, std=1.0 / (self.cfg.hidden_size**0.5))
            block.scale_shift_table.initialized = True

            if self.cfg.qk_norm:
                nn.init.ones_(block.attn.q_norm.weight)
                nn.init.ones_(block.attn.k_norm.weight)
                block.attn.q_norm.weight.initialized = True
                block.attn.k_norm.weight.initialized = True

            if self.cfg.cross_norm:
                nn.init.ones_(block.cross_attn.q_norm.weight)
                nn.init.ones_(block.cross_attn.k_norm.weight)
                block.cross_attn.q_norm.weight.initialized = True
                block.cross_attn.k_norm.weight.initialized = True

        nn.init.normal_(self.final_layer.scale_shift_table, mean=0.0, std=1.0 / (self.cfg.hidden_size**0.5))
        self.final_layer.scale_shift_table.initialized = True

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in [
                "x_embedder",
                "t_embedder",
                "t_block",
                "y_embedder",
                "attention_y_norm",
                "blocks",
                "final_layer",
            ]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)
        for name, parameter in self.named_parameters(recurse=False):
            if name in ["pos_embed"]:
                pass
            else:
                raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=False)
        if self.cfg.pretrained_source in ["sana", "sana_t2i"]:
            self.load_state_dict(checkpoint["state_dict"])
        elif self.cfg.pretrained_source == "dc-ae":
            if "ema" in checkpoint:
                checkpoint = next(iter(checkpoint["ema"].values()))
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            new_ckpt = {}
            for key in checkpoint.keys():
                if "pos_embed" in key:
                    continue
                new_ckpt["0." + key] = checkpoint[key]
            self.get_trainable_modules_list().load_state_dict(new_ckpt)
        elif self.cfg.pretrained_source == "dc-ae-fsdp":
            self.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    def enable_activation_checkpointing(self, mode: str):
        if mode == "transformer":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

            for i in range(len(self.blocks)):
                self.blocks[i] = checkpoint_wrapper(self.blocks[i], preserve_rng_state=False)
        else:
            raise ValueError(f"mode {mode} is not supported")

    def unpatchify(self, x, c, h, w):
        p = self.x_embedder.patch_size
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_without_cfg(self, x, t, y, mask):
        assert mask is not None
        t = t.long()
        height, width = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        in_channels = x.shape[1]

        y = y.to(self.y_embedder.y_proj.fc1.weight.dtype)
        x = x.to(y.dtype)

        x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)

        t0 = self.t_block(t)

        y = self.y_embedder(y, mask=mask)  # (N, D)
        y = self.attention_y_norm(y)

        if mask.shape[0] != y.shape[0]:
            if y.shape[0] % mask.shape[0] == 0:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            else:
                raise ValueError(f"First channel of mask must be a factor of the first channel of y.")

        for block in self.blocks:
            x = block(x, y, t0, mask, (height, width))  # (N, T, D) #support grad checkpoint

        x = self.final_layer(x, t, out_channels=in_channels)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, in_channels, height, width)  # (N, out_channels, H, W)

        return x

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        text_embeddings = text_embed_info[self.cfg.text_encoder_name]["text_embeddings"]
        text_embedding_masks = text_embed_info[self.cfg.text_encoder_name]["text_embedding_masks"]
        null_text_embeddings = self.y_embedder.y_embedding.repeat(text_embeddings.shape[0], 1, 1).unsqueeze(1)

        device = get_device(self)
        bs = text_embeddings.shape[0]

        if noise is None:
            z = torch.randn(
                bs,
                self.cfg.in_channels,
                self.cfg.input_size,
                self.cfg.input_size,
                device=device,
                generator=generator,
            )
        else:
            z = noise

        model_kwargs = dict(mask=text_embedding_masks)

        if self.cfg.eval_scheduler == "DPMS":
            from ....c2icore.diffusioncore.models.sana_utils.dpm_solver import DPMS

            dpm_solver = DPMS(
                self.forward_without_cfg,
                condition=text_embeddings,
                uncondition=null_text_embeddings,
                guidance_type=self.cfg.guidance_type,
                cfg_scale=cfg_scale,
                pag_scale=pag_scale,
                pag_applied_layers=self.cfg.pag_applied_layers,
                model_type="flow",
                model_kwargs=model_kwargs,
                schedule="FLOW",
                interval_guidance=self.cfg.interval_guidance,
            )
            samples = dpm_solver.sample(
                z,
                steps=self.cfg.num_inference_steps,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=self.cfg.flow_shift,
            )
        else:
            raise ValueError(f"Eval scheduler {self.cfg.eval_scheduler} is not supported.")

        return samples

    def forward(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        y = text_embed_info[self.cfg.text_encoder_name]["text_embeddings"]
        mask = text_embed_info[self.cfg.text_encoder_name]["text_embedding_masks"]

        info = {}
        detailed_loss_dict = {}
        device = x.device

        if self.cfg.train_scheduler == "SanaScheduler":
            from ....c2icore.diffusioncore.models.sana_utils.sana_sampler import (
                compute_density_for_timestep_sampling,
            )

            bs = x.shape[0]
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bs,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=None,  # not used
            )
            t = (u * self.cfg.train_sampling_steps).long().to(device)
            scheduler_output = self.train_scheduler.training_losses(
                self.forward_without_cfg, x, t, model_kwargs=dict(y=y, mask=mask)
            )
            loss = scheduler_output["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        detailed_loss_dict["loss"] = loss.item()
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
