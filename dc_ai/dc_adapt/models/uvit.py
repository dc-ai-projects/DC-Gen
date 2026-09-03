# UViT was introduced by Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu in "All are Worth Words: A ViT Backbone for Diffusion Models", see https://arxiv.org/abs/2209.12152.
# The original implementation is by Fan Bao, licensed under the MIT License. See https://github.com/baofff/U-ViT/blob/main/libs/uvit.py.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...apps.utils.init import init_modules
from ...c2icore.models.ops.input_embed import AdaptivePatchEmbed
from .base import BasePatchEmbedding, BasePatchEmbeddingConfig

__all__ = ["UViTPatchEmbedding"]


@dataclass
class UViTPatchEmbeddingConfig(BasePatchEmbeddingConfig):
    patch_kernel_size: Optional[int] = None


class UViTPatchEmbedding(BasePatchEmbedding):
    def __init__(self, cfg: UViTPatchEmbeddingConfig):
        super().__init__(cfg)
        self.cfg: UViTPatchEmbeddingConfig

        assert self.patch_ffn is None

    def build_x_embedder(self) -> None:
        self.patch_embed = AdaptivePatchEmbed(
            self.cfg.input_size,
            self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.hidden_size,
            bias=True,
            share_weights=False,
            kernel_size=self.cfg.patch_kernel_size,
        )
        num_patches = (self.cfg.input_size // self.cfg.patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.zeros(1, 2 + num_patches, self.cfg.hidden_size))

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        patch_embedding = {}
        for name, module in self.named_children():
            if name in ["patch_embed"]:
                patch_embedding[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        patch_embedding = nn.ModuleDict(patch_embedding)
        for name, parameter in self.named_parameters(recurse=False):
            if name in ["pos_embed"]:
                setattr(patch_embedding, name, parameter)
            else:
                raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(patch_embedding)
        return nn.ModuleList(trainable_modules_list)

    def initialize_weights(self) -> None:
        init_modules(self, init_type="trunc_normal@0.02")
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_embed.initialized = True
        super().initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x, keep_2d=True)
        H, W = x.shape[-2:]
        pos_embed = self.pos_embed[:, 2:, :].permute(0, 2, 1).contiguous().view(1, -1, H, W)
        x = x + pos_embed
        return x
