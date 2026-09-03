# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/builder.py and https://github.com/NVlabs/Sana.
# This implementation is a modified version for class-to-image generation.

from dataclasses import dataclass

import torch
from torch import nn

from ...t2icore.models.ops.input_embed import PatchEmbedMS
from .base import BasePatchEmbedding, BasePatchEmbeddingConfig


@dataclass
class SanaT2IPatchEmbeddingConfig(BasePatchEmbeddingConfig):
    pass


class SanaT2IPatchEmbedding(BasePatchEmbedding):
    def __init__(self, cfg: SanaT2IPatchEmbeddingConfig):
        super().__init__(cfg)
        self.cfg: SanaT2IPatchEmbeddingConfig

    def build_x_embedder(self):
        self.x_embedder = PatchEmbedMS(
            patch_size=self.cfg.patch_size,
            in_channels=self.cfg.in_channels,
            embed_dim=self.cfg.hidden_size,
            bias=True,
        )

    def initialize_weights(self) -> None:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.x_embedder.proj.weight.initialized = True
        self.x_embedder.proj.bias.initialized = True
        super().initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]
        x = self.x_embedder(x)
        if self.patch_ffn is not None:
            x = self.patch_ffn(x)
        x = x.unflatten(1, (H, W)).permute(0, 3, 1, 2)
        return x
