# DiT was introduced by William Peebles and Saining Xie in "Scalable Diffusion Models with Transformers", see http://arxiv.org/abs/2212.09748.
# The original implementation is by Meta Platforms, Inc. and affiliates, licensed under CC-BY-NC. See https://github.com/facebookresearch/DiT.

from dataclasses import dataclass

import torch
import torch.nn as nn

from ...apps.utils.init import init_modules
from ...c2icore.diffusioncore.models.dit import get_2d_sincos_pos_embed
from ...c2icore.models.ops.input_embed import AdaptivePatchEmbed
from .base import BasePatchEmbedding, BasePatchEmbeddingConfig

__all__ = ["DiTPatchEmbedding"]


@dataclass
class DiTPatchEmbeddingConfig(BasePatchEmbeddingConfig):
    pass


class DiTPatchEmbedding(BasePatchEmbedding):
    def __init__(self, cfg: DiTPatchEmbeddingConfig):
        super().__init__(cfg)
        self.cfg: DiTPatchEmbeddingConfig

    def build_x_embedder(self) -> None:
        self.x_embedder = AdaptivePatchEmbed(
            self.cfg.input_size,
            self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.hidden_size,
            bias=True,
            share_weights=False,
            kernel_size=self.cfg.patch_kernel_size,
        )
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, self.cfg.hidden_size), requires_grad=False
        )

    def initialize_weights(self) -> None:
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.cfg.patch_kernel_size is not None:
            init_modules(self.x_embedder, init_type="trunc_normal@0.02")
        else:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)
            self.x_embedder.proj.weight.initialized = True
            self.x_embedder.proj.bias.initialized = True

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.initialized = True

        super().initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.x_embedder(x, keep_2d=True)
        if self.patch_ffn is not None:
            x = self.patch_ffn(x)
        H, W = x.shape[-2:]
        pos_embed = self.pos_embed.permute(0, 2, 1).contiguous().view(1, -1, H, W)
        x = x + pos_embed
        return x
