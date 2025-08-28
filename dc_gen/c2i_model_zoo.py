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

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import MISSING, OmegaConf

from .c2icore.autoencoder import Autoencoder, AutoencoderConfig
from .c2icore.diffusioncore.models.dit import DiT, DiTConfig, dc_ae_dit_xl_in_512px, dc_ae_sit_xl_in_512px
from .c2icore.diffusioncore.models.uvit import (
    UViT,
    UViTConfig,
    dc_ae_1_5_usit_2b_in_512px,
    dc_ae_1_5_usit_3b_in_512px,
    dc_ae_usit_2b_in_512px,
    dc_ae_usit_h_in_512px,
    dc_ae_uvit_2b_in_512px,
    dc_ae_uvit_h_in_512px,
    dc_ae_uvit_s_in_512px,
)

__all__ = ["DCAE_Diffusion_HF"]


@dataclass
class DCAEC2IPipelineConfig:
    resolution: int = 512
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    model: str = MISSING
    dit: DiTConfig = field(default_factory=DiTConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)


REGISTERED_DCAE_DIFFUSION_MODEL: dict[
    str, tuple[Callable[[str, float, int, Optional[str]], str], str, float, int, Optional[str]]
] = {
    "dc-ae-f32c32-in-1.0-dit-xl-in-512px": (
        dc_ae_dit_xl_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-dit-xl-in-512px-trainbs1024": (
        dc_ae_dit_xl_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f32c32-in-1.0-uvit-s-in-512px": (
        dc_ae_uvit_s_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-uvit-h-in-512px": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-uvit-2b-in-512px": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f32c32-in-1.0-usit-h-in-512px": (
        dc_ae_usit_h_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    "dc-ae-f32c32-in-1.0-usit-2b-in-512px": (
        dc_ae_usit_2b_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f64c128-in-1.0-uvit-h-in-512px": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-h-in-512px-train2000k": (
        dc_ae_uvit_h_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-2b-in-512px": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    "dc-ae-f64c128-in-1.0-uvit-2b-in-512px-train2000k": (
        dc_ae_uvit_2b_in_512px,
        "dc-ae-f64c128-in-1.0",
        0.2889,
        128,
        None,
    ),
    ################################################################################
    "dc-ae-f32c32-in-1.0-sit-xl-in-512px": (
        dc_ae_sit_xl_in_512px,
        "dc-ae-f32c32-in-1.0",
        0.3189,
        32,
        None,
    ),
    ################################################################################
    "dc-ae-f64c128-1.5-usit-2b-in-512px": (
        dc_ae_1_5_usit_2b_in_512px,
        "dc-ae-f64c128-1.5",
        0.5935,
        128,
        None,
    ),
    "dc-ae-f64c128-1.5-usit-3b-in-512px": (
        dc_ae_1_5_usit_3b_in_512px,
        "dc-ae-f64c128-1.5",
        0.5935,
        128,
        None,
    ),
}


def create_dc_ae_diffusion_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEC2IPipelineConfig:
    diffusion_cls, ae_name, scaling_factor, in_channels, default_pt = REGISTERED_DCAE_DIFFUSION_MODEL[name]
    pretrained_path = default_pt if pretrained_path is None else pretrained_path
    cfg_str = diffusion_cls(ae_name, scaling_factor, in_channels, pretrained_path)
    cfg: DCAEC2IPipelineConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DCAEC2IPipelineConfig), OmegaConf.from_dotlist(cfg_str.split(" ")))
    )
    return cfg


class DCAE_Diffusion_HF(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_name: str):
        super().__init__()
        cfg = create_dc_ae_diffusion_model_cfg(model_name)
        self._autoencoder = [Autoencoder(cfg.autoencoder)]
        if cfg.model == "dit":
            cfg.dit.input_size = cfg.resolution // self.autoencoder.spatial_compression_ratio
            self.diffusion_model = DiT(cfg.dit)
        elif cfg.model == "uvit":
            cfg.uvit.input_size = cfg.resolution // self.autoencoder.spatial_compression_ratio
            self.diffusion_model = UViT(cfg.uvit)
        else:
            raise ValueError(f"model {cfg.model} is not supported")

    @property
    def autoencoder(self):
        return self._autoencoder[0]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._autoencoder[0].to(*args, **kwargs)
        return self
