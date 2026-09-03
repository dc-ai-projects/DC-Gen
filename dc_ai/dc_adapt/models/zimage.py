# Copyright 2025 Alibaba Z-Image Team and NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping

import torch
import torch.nn as nn
from einops import rearrange

from ...models.utils import get_submodule_weights
from ...t2icore.models.zimage import ZImageConfig

__all__ = ["ZImagePatchEmbedding"]


class ZImagePatchEmbedding(nn.Module):
    """
    ZImage Patch Embedding module for DC-Adapt training.
    This module handles the initial patch embedding layer that needs to be adapted
    when switching from flux-vae (16 channels) to dc-ae (32 channels).
    """

    def __init__(self, cfg: ZImageConfig):
        super(ZImagePatchEmbedding, self).__init__()
        self.cfg = cfg

        self._build_model()
        self.initialize_weights()
        if cfg.pretrained_path is not None:
            self.load_model()

    def _build_model(self) -> None:
        """
        Build the patch embedding layer.
        Similar to ZImage's patchify_and_embed, this converts image patches to embeddings.
        ZImage uses Linear layer for patch embedding (f_patch_size=1 for images).
        """
        # For DC-Adapt, we assume f_patch_size=1 (image, not video)
        # input (after rearrange): [B, num_patches, patch_size*patch_size*in_channels]
        # output: [B, num_patches, dim]
        self.x_embedder = nn.Linear(
            self.cfg.patch_size * self.cfg.patch_size * self.cfg.in_channels, self.cfg.dim, bias=True
        )

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["x_embedder"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self) -> None:
        """Load pretrained weights from ZImage checkpoint."""
        if self.cfg.pretrained_path is None:
            raise ValueError("Z-Image patch embedding loading requires pretrained_path")

        checkpoint = torch.load(
            self.cfg.pretrained_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Z-Image checkpoint must contain a state dict: {self.cfg.pretrained_path}")

        if self.cfg.pretrained_source == "diffusers":
            state_dict = checkpoint
        elif self.cfg.pretrained_source == "dc-gen":
            model_state_dict = checkpoint.get("model_state_dict")
            if not isinstance(model_state_dict, Mapping):
                raise ValueError(
                    "Z-Image DC-Gen checkpoint must contain model_state_dict: " f"{self.cfg.pretrained_path}"
                )
            state_dict = get_submodule_weights(model_state_dict, "transformer.")
            if not state_dict:
                raise ValueError(
                    "Z-Image DC-Gen checkpoint model_state_dict must contain transformer.* weights: "
                    f"{self.cfg.pretrained_path}"
                )
        else:
            raise ValueError(f"Unsupported Z-Image pretrained source {self.cfg.pretrained_source!r}")

        embedder_prefix = f"all_x_embedder.{self.cfg.patch_size}-1."
        x_embedder_state = get_submodule_weights(state_dict, embedder_prefix)
        if not x_embedder_state:
            raise ValueError(
                f"Z-Image checkpoint does not contain {embedder_prefix}* weights: {self.cfg.pretrained_path}"
            )
        self.x_embedder.load_state_dict(x_embedder_state, strict=True)

    def initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        # Initialize x_embedder Linear layer
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Mark as initialized (required by dc-gen framework)
        self.x_embedder.weight.initialized = True
        self.x_embedder.bias.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Embedded patches of shape [B, dim, H/patch_size, W/patch_size]
        """
        bsz, num_channels, height, width = x.shape

        # Rearrange from image to patches
        # [B, C, H, W] -> [B, num_patches, patch_size*patch_size*C]
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.cfg.patch_size, p2=self.cfg.patch_size)

        # Apply linear embedding
        x = self.x_embedder(x)

        # Rearrange back to spatial format
        # [B, num_patches, dim] -> [B, dim, H/patch_size, W/patch_size]
        x = rearrange(x, "b (h w) c -> b c h w", h=height // self.cfg.patch_size)

        return x
