# UViT was introduced by Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu in "All are Worth Words: A ViT Backbone for Diffusion Models", see https://arxiv.org/abs/2209.12152.
# The original implementation is by Fan Bao, licensed under the MIT License. See https://github.com/baofff/U-ViT/blob/main/libs/uvit.py.

from dataclasses import dataclass
from functools import partial
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.vision_transformer import Mlp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn import functional as F

from ....apps.utils.dist import is_master
from ....models.utils.network import get_submodule_weights
from ...models.ops.input_embed import AdaptivePatchEmbed
from ...models.ops.label_embed import LabelEmbedder
from ...models.ops.timestep_embed import timestep_embedding
from .base import BaseDiffusionModel, BaseDiffusionModelConfig

__all__ = ["UViTConfig", "UViT", "dc_ae_uvit_s_in_512px", "dc_ae_uvit_h_in_512px"]


if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    ATTENTION_MODE = "flash"
else:
    ATTENTION_MODE = "math"


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, "B C (h p1) (w p2) -> B (h w) (p1 p2 C)", p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1] and patch_size**2 * channels == x.shape[2]
    x = einops.rearrange(x, "B (h w) (p1 p2 C) -> B C (h p1) (w p2)", h=h, p1=patch_size, p2=patch_size)
    return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, proj_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5 if qk_scale is None else qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == "flash_attn":
            from flash_attn import flash_attn_qkvpacked_func

            qkv = einops.rearrange(qkv, "B L (K H D) -> B L K H D", K=3, H=self.num_heads)
            x = flash_attn_qkvpacked_func(qkv, softmax_scale=self.scale)
            x = einops.rearrange(x, "B L H D -> B L (H D)")
        elif ATTENTION_MODE == "flash":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, "B H L D -> B L (H D)")
        elif ATTENTION_MODE == "math":
            qkv = einops.rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        skip=False,
    ):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None

    def forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AdaptiveLinear(nn.Linear):
    def forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        x = F.linear(x, self.weight[:out_channels], self.bias[:out_channels])
        return x


@dataclass
class UViTConfig(BaseDiffusionModelConfig):
    name: str = "UViT"

    patch_size: int = 2
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    mlp_time_embed: bool = False
    qkv_bias: bool = False
    act_layer: str = "gelu"
    patch_kernel_size: Optional[int] = None

    class_dropout_prob: float = 0.1
    num_classes: int = 1000

    eval_scheduler: str = "DPM_Solver"
    num_inference_steps: int = 30
    train_scheduler: str = "DPM_Solver"

    attn_mode: Optional[str] = None

    freeze_backbone: bool = False
    head_only: bool = False


class UViT(BaseDiffusionModel):
    def __init__(self, cfg: UViTConfig):
        super().__init__(cfg)
        self.cfg: UViTConfig

        if cfg.attn_mode is not None:
            global ATTENTION_MODE
            ATTENTION_MODE = cfg.attn_mode
        if is_master():
            print(f"attention mode is {ATTENTION_MODE}")

        if cfg.freeze_backbone or cfg.head_only:
            # freeze all parameters
            for parameter in self.parameters():
                parameter.requires_grad = False
            if cfg.head_only:
                unfreeze_modules = [self.decoder_pred, self.final_layer]
            else:
                unfreeze_modules = [self.patch_embed, self.norm, self.decoder_pred, self.final_layer]
            # unfreeze the parameters of selected modules
            for m in unfreeze_modules:
                if m is not None:
                    for parameter in m.parameters():
                        parameter.requires_grad = True

    def build_model(self):
        self.patch_embed = AdaptivePatchEmbed(
            self.cfg.input_size,
            self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.hidden_size,
            bias=True,
            share_weights=self.cfg.adaptive_channel_share_weights,
            kernel_size=self.cfg.patch_kernel_size,
        )
        num_patches = (self.cfg.input_size // self.cfg.patch_size) ** 2

        self.time_embed = (
            nn.Sequential(
                nn.Linear(self.cfg.hidden_size, 4 * self.cfg.hidden_size),
                nn.SiLU(),
                nn.Linear(4 * self.cfg.hidden_size, self.cfg.hidden_size),
            )
            if self.cfg.mlp_time_embed
            else nn.Identity()
        )

        if self.cfg.num_classes > 0:
            self.label_emb = LabelEmbedder(
                self.cfg.num_classes, self.cfg.hidden_size, self.cfg.class_dropout_prob
            )  # nn.Embedding(self.cfg.num_classes, self.cfg.hidden_size)
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, self.extras + num_patches, self.cfg.hidden_size))

        if self.cfg.act_layer == "gelu":
            act_layer = nn.GELU
        elif self.cfg.act_layer == "silu":
            act_layer = partial(nn.SiLU, inplace=True)
        else:
            raise NotImplementedError(f"act_layer {act_layer} is not supported")

        self.in_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.cfg.hidden_size,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    qkv_bias=self.cfg.qkv_bias,
                    qk_scale=None,
                    act_layer=act_layer,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(self.cfg.depth // 2)
            ]
        )

        self.mid_block = Block(
            dim=self.cfg.hidden_size,
            num_heads=self.cfg.num_heads,
            mlp_ratio=self.cfg.mlp_ratio,
            qkv_bias=self.cfg.qkv_bias,
            qk_scale=None,
            act_layer=act_layer,
            norm_layer=nn.LayerNorm,
        )

        self.out_blocks = nn.ModuleList(
            [
                Block(
                    dim=self.cfg.hidden_size,
                    num_heads=self.cfg.num_heads,
                    mlp_ratio=self.cfg.mlp_ratio,
                    qkv_bias=self.cfg.qkv_bias,
                    qk_scale=None,
                    act_layer=act_layer,
                    norm_layer=nn.LayerNorm,
                    skip=True,
                )
                for _ in range(self.cfg.depth // 2)
            ]
        )

        self.norm = nn.LayerNorm(self.cfg.hidden_size)
        self.patch_dim = self.cfg.patch_size**2 * self.cfg.in_channels
        if self.cfg.adaptive_channel:
            self.decoder_pred = AdaptiveLinear(self.cfg.hidden_size, self.patch_dim, bias=True)
        else:
            self.decoder_pred = nn.Linear(self.cfg.hidden_size, self.patch_dim, bias=True)
        self.final_layer = nn.Identity()

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in [
                "patch_embed",
                "time_embed",
                "label_emb",
                "in_blocks",
                "mid_block",
                "out_blocks",
                "norm",
                "decoder_pred",
                "final_layer",
            ]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)
        for name, parameter in self.named_parameters(recurse=False):
            if name in ["pos_embed"]:
                setattr(diffusion_model, name, parameter)
            else:
                raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "uvit":
            if "ema" in checkpoint:
                checkpoint = checkpoint["ema"]
            self.patch_embed.load_state_dict(get_submodule_weights(checkpoint, "patch_embed."))
            self.time_embed.load_state_dict(get_submodule_weights(checkpoint, "time_embed."))
            if self.cfg.num_classes > 0:
                self.label_emb.embedding_table.load_state_dict(get_submodule_weights(checkpoint, "label_emb."))
            self.pos_embed.data = checkpoint["pos_embed"]
            self.in_blocks.load_state_dict(get_submodule_weights(checkpoint, "in_blocks."))
            self.mid_block.load_state_dict(get_submodule_weights(checkpoint, "mid_block."))
            self.out_blocks.load_state_dict(get_submodule_weights(checkpoint, "out_blocks."))
            self.norm.load_state_dict(get_submodule_weights(checkpoint, "norm."))
            self.decoder_pred.load_state_dict(get_submodule_weights(checkpoint, "decoder_pred."))
            self.final_layer.load_state_dict(get_submodule_weights(checkpoint, "final_layer."))
        elif self.cfg.pretrained_source == "dc-adapt":
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
            self.load_state_dict(get_submodule_weights(checkpoint, "uvit."))
        elif self.cfg.pretrained_source == "dc-ae-fsdp":
            checkpoint = next(iter(checkpoint["ema"].values()))
            self.load_state_dict(checkpoint)
        else:
            raise NotImplementedError(f"pretrained source {self.cfg.pretrained_source} is not supported")

    def _init_weights(self, module: nn.Module):
        if all(param is None for param in module.parameters(recurse=False)):  # no parameter at this level
            pass
        elif isinstance(module, UViT):  # pos_embed is separately initialized
            pass
        elif isinstance(
            module, (nn.Conv2d, nn.Embedding)
        ):  # UViT doesn't have custom initialization. Don't copy this to other models
            for param in module.parameters():
                param.initialized = True
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            module.weight.initialized = True
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
                module.bias.initialized = True
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            module.weight.initialized = True
            module.bias.initialized = True
        else:
            raise ValueError(f"module {module} is not supported")

    def initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_embed.initialized = True
        self.apply(self._init_weights)

    def enable_activation_checkpointing(self, mode: str):
        if mode == "full":
            for i in range(len(self.in_blocks)):
                self.in_blocks[i] = checkpoint_wrapper(self.in_blocks[i], preserve_rng_state=False)
            self.mid_block = checkpoint_wrapper(self.mid_block, preserve_rng_state=False)
            for i in range(len(self.out_blocks)):
                self.out_blocks[i] = checkpoint_wrapper(self.out_blocks[i], preserve_rng_state=False)
        elif mode.startswith("first_"):
            num_blocks = int(mode.split("_")[1])
            for i in range(len(self.in_blocks)):
                if num_blocks > 0:
                    self.in_blocks[i] = checkpoint_wrapper(self.in_blocks[i], preserve_rng_state=False)
                    num_blocks -= 1
            if num_blocks > 0:
                self.mid_block = checkpoint_wrapper(self.mid_block, preserve_rng_state=False)
                num_blocks -= 1
            for i in range(len(self.out_blocks)):
                if num_blocks > 0:
                    self.out_blocks[i] = checkpoint_wrapper(self.out_blocks[i], preserve_rng_state=False)
                    num_blocks -= 1
        else:
            raise ValueError(f"mode {mode} is not supported")

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        info = {}
        if self.cfg.count_nfe:
            self.nfe += 1
        in_channels = x.shape[1]
        x = self.patch_embed(x)
        _, L, _ = x.shape

        time_token = self.time_embed(timestep_embedding(t, self.cfg.hidden_size))
        time_token = time_token.unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)
        if y is not None and self.cfg.num_classes > 0:
            # y is None only when the model is trained for unconditional sampling
            # for classifier-free guidance, the null type will be assigned a valid label
            label_emb = self.label_emb(y, self.training)
            label_emb = label_emb.unsqueeze(dim=1)
            x = torch.cat((label_emb, x), dim=1)
        x = x + self.pos_embed

        skips = []
        for i, blk in enumerate(self.in_blocks):
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        if self.cfg.head_only:
            x = x.detach()

        x = self.norm(x)
        if self.cfg.adaptive_channel:
            x = self.decoder_pred(x, out_channels=in_channels * self.cfg.patch_size**2)
        else:
            x = self.decoder_pred(x)
        assert x.size(1) == self.extras + L
        x = x[:, self.extras :, :]
        x = unpatchify(x, in_channels)
        x = self.final_layer(x)
        return x, info


def dc_ae_uvit_s_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=12 uvit.hidden_size=512 uvit.num_heads=8 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_uvit_h_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_uvit_2b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_usit_h_in_512px(ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels={in_channels} uvit.patch_size=1 "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_usit_2b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path}"
    )


def dc_ae_1_5_usit_2b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=28 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 uvit.adaptive_channel=True "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} uvit.pretrained_source=dc-ae-fsdp"
    )


def dc_ae_1_5_usit_3b_in_512px(
    ae_name: str, scaling_factor: float, in_channels: int, pretrained_path: Optional[str]
) -> str:
    return (
        f"autoencoder.name={ae_name} autoencoder.scaling_factor={scaling_factor} "
        f"model=uvit uvit.depth=56 uvit.hidden_size=2048 uvit.num_heads=32 uvit.in_channels={in_channels} uvit.patch_size=1 uvit.adaptive_channel=True "
        "uvit.train_scheduler=SiTSampler uvit.eval_scheduler=ODE_dopri5 uvit.num_inference_steps=250 "
        f"uvit.pretrained_path={'null' if pretrained_path is None else pretrained_path} uvit.pretrained_source=dc-ae-fsdp"
    )
