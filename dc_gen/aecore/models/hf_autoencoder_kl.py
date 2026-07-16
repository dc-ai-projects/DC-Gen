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
class HFAutoencoderKLConfig(BaseAEConfig):
    model_name: str = "stabilityai/sd-vae-ft-ema"


class HFAutoencoderKL(BaseAE):
    def __init__(self, cfg: HFAutoencoderKLConfig):
        super().__init__(cfg)
        import diffusers

        self.cfg: HFAutoencoderKLConfig
        if cfg.model_name in ["stabilityai/sd-vae-ft-ema", "zelaki/eq-vae", "zelaki/eq-vae-ema"]:
            self.autoencoder = diffusers.models.AutoencoderKL.from_pretrained(cfg.model_name)
        elif cfg.model_name == "flux-vae":
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            self.autoencoder = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
        else:
            raise ValueError(f"model {cfg.model_name} is not supported")

    def encode(self, x: torch.Tensor, latent_channels: Optional[list[int]] = None) -> torch.Tensor | list[torch.Tensor]:
        posterior = self.autoencoder.encode(x).latent_dist
        return posterior.sample()

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        output = self.autoencoder.decode(x).sample
        return output
