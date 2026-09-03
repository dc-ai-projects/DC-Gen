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

from .base import BaseAE, BaseAEConfig


@dataclass
class DCAEDiffusersConfig(BaseAEConfig):
    model_name: str = "dc-ae-f32c32-sana-1.0"


class DCAEDiffusers(BaseAE):
    def __init__(self, cfg: DCAEDiffusersConfig):
        super().__init__(cfg)
        from diffusers import AutoencoderDC

        self.cfg: DCAEDiffusersConfig
        self.model: AutoencoderDC = AutoencoderDC.from_pretrained(f"mit-han-lab/{cfg.model_name}-diffusers")

    def encode(self, x: torch.Tensor, latent_channels: Optional[list[int]] = None) -> torch.Tensor | list[torch.Tensor]:
        return self.model.encode(x).latent

    def decode(self, x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        return self.model.decode(x).sample
