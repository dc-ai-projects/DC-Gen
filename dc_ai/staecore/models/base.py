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
from torch import nn


@dataclass
class BaseSTAECoreModelConfig:
    pass


class BaseSTAECoreModel(nn.Module):
    def __init__(self, cfg: BaseSTAECoreModelConfig):
        super().__init__()
        self.cfg = cfg

    @property
    def spatial_compression_ratio(self) -> int:
        raise NotImplementedError

    @property
    def temporal_compression_ratio(self) -> int:
        raise NotImplementedError

    # for video reconstruction, T should be k * temporal_divisor + temporal_remainder
    @property
    def temporal_divisor(self) -> int:
        return self.temporal_compression_ratio

    @property
    def temporal_remainder(self) -> int:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def reconstruct_video(self, x: torch.Tensor, *, latent_channels: Optional[int] = None) -> tuple[torch.Tensor, dict]:
        """
        x: (B, 3, T, H, W) [-1, 1]
        """
        num_frames = x.shape[2]
        padding_frames = (self.temporal_remainder - num_frames) % self.temporal_divisor
        if padding_frames != 0:
            print(f"padding {padding_frames} frames to the end")
            padding = x[:, :, -1:].repeat(1, 1, padding_frames, 1, 1)
            x = torch.cat((x, padding), dim=2)
            num_frames = x.size(2)
        z = self.encode(x)
        if latent_channels is not None:
            z = z * (torch.arange(z.shape[1], device=z.device) < latent_channels)[:, None, None, None]
        y = self.decode(z)
        if padding_frames > 0:
            y = y[:, :, :-padding_frames]
        return y, {"latent": z}

    def forward(self, func_name: str, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)
