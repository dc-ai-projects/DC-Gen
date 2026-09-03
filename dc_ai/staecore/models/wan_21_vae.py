# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from .base import BaseSTAECoreModel, BaseSTAECoreModelConfig


@dataclass
class Wan21VAEConfig(BaseSTAECoreModelConfig):
    model_id: str = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


class Wan21VAE(BaseSTAECoreModel):
    def __init__(self, cfg: Wan21VAEConfig):
        super().__init__(cfg)
        self.cfg: Wan21VAEConfig

        from diffusers import AutoencoderKLWan

        self.model = AutoencoderKLWan.from_pretrained(cfg.model_id, subfolder="vae", torch_dtype=torch.float32)

    @property
    def spatial_compression_ratio(self) -> int:
        return 8

    @property
    def temporal_compression_ratio(self) -> int:
        return 4

    @property
    def temporal_remainder(self) -> int:
        return 1

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x).latent_dist.sample()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z).sample

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        y, info = self.reconstruct_video(x[:, :, None])
        return y[:, :, 0], info
