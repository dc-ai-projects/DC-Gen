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
from typing import Optional

import torch
import torch.nn as nn

__all__ = ["BaseAEConfig", "BaseAE"]


@dataclass
class BaseAEConfig:
    pass


class BaseAE(nn.Module):
    def __init__(self, cfg: BaseAEConfig):
        super().__init__()
        self.cfg = cfg

    @property
    def spatial_compression_ratio(self) -> int:
        raise NotImplementedError

    def encode(self, x: torch.Tensor, latent_channels: Optional[list[int]] = None) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def decode(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, global_step: int, latent_channels: Optional[list[int]] = None) -> torch.Tensor:
        raise NotImplementedError

    def reconstruct_image(self, x: torch.Tensor, latent_channels: Optional[list[int]] = None) -> torch.Tensor:
        """
        image: (B, 3, H, W) [-1, 1]
        """
        latent = self.encode(x, latent_channels=latent_channels)
        y = self.decode(latent)
        return y, {}, {"latent": latent}
