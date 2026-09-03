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
from typing import Any, Optional

import numpy as np
import torch
from torch import nn

from ..models.utils.list import val2list
from ..stae_model_zoo import REGISTERED_DCAEV_MODEL
from .models.base import BaseSTAECoreModel
from .models.dc_ae_v import DCAEV


class SingleAutoencoder(nn.Module):
    model_dict: dict[
        str, tuple[BaseSTAECoreModel, int, int, int, int, Optional[float | np.ndarray], Optional[float | np.ndarray]]
    ] = {}

    @classmethod
    def build_model(cls, model_name: str):
        if model_name in REGISTERED_DCAEV_MODEL:
            if REGISTERED_DCAEV_MODEL[model_name][1] is None:
                raise NotImplementedError
            else:
                st_dc_ae_cfg = REGISTERED_DCAEV_MODEL[model_name][0](model_name, REGISTERED_DCAEV_MODEL[model_name][1])
                model = DCAEV(st_dc_ae_cfg)
            latent_channels = model.cfg.latent_channels
            scaling_factor = None
            shifting_factor = None
        elif model_name == "wan-vae":
            from .models.wan_21_vae import Wan21VAE, Wan21VAEConfig

            model = Wan21VAE(Wan21VAEConfig())
            latent_channels = 16
            shifting_factor = -np.array(model.model.config.latents_mean)
            scaling_factor = 1 / np.array(model.model.config.latents_std)
        elif model_name == "wan-2.2-vae":
            from .models.wan_22_vae import Wan22VAE, Wan22VAEConfig

            model = Wan22VAE(Wan22VAEConfig())
            latent_channels = 48
            shifting_factor = -model.mean
            scaling_factor = 1 / model.std
        else:
            raise ValueError(f"autoencoder {model_name} is not supported")
        spatial_compression_ratio = model.spatial_compression_ratio
        temporal_compression_ratio = model.temporal_compression_ratio
        temporal_remainder = model.temporal_remainder

        cls.model_dict[model_name] = (
            model.eval(),
            spatial_compression_ratio,
            temporal_compression_ratio,
            temporal_remainder,
            latent_channels,
            scaling_factor,
            shifting_factor,
        )

    def __init__(
        self,
        model_name: str,
        scaling_factor: Optional[float | np.ndarray],
        shifting_factor: Optional[float | np.ndarray],
        latent_channels: Optional[int],
    ):
        super().__init__()
        self.model_name = model_name
        if model_name not in SingleAutoencoder.model_dict:
            self.build_model(model_name)
        (
            self.model,
            self.spatial_compression_ratio,
            self.temporal_compression_ratio,
            self.temporal_remainder,
            self.default_latent_channels,
            default_scaling_factor,
            default_shifting_factor,
        ) = SingleAutoencoder.model_dict[model_name]
        if scaling_factor is None:
            assert default_scaling_factor is not None
            scaling_factor = default_scaling_factor
        self.scaling_factor = scaling_factor
        if shifting_factor is None:
            shifting_factor = default_shifting_factor
        self.shifting_factor = shifting_factor
        if latent_channels is None:
            latent_channels = self.default_latent_channels
        self.latent_channels = latent_channels
        if isinstance(scaling_factor, np.ndarray):
            assert scaling_factor.shape[0] >= latent_channels
        if isinstance(shifting_factor, np.ndarray):
            assert shifting_factor.shape[0] >= latent_channels

    def apply_scaling_and_shifting_after_encode(self, latent: torch.Tensor) -> torch.Tensor:
        assert latent.shape[1] == self.latent_channels

        if self.shifting_factor is None:
            pass
        elif isinstance(self.shifting_factor, float):
            latent = latent + self.shifting_factor
        elif isinstance(self.shifting_factor, np.ndarray) and self.latent_channels <= self.shifting_factor.shape[0]:
            latent = (
                latent
                + torch.tensor(self.shifting_factor, dtype=latent.dtype, device=latent.device)[
                    (None, slice(self.latent_channels)) + (None,) * (latent.ndim - 2)
                ]
            )
        else:
            raise ValueError(f"shifting_factor {self.shifting_factor} is not supported")

        if isinstance(self.scaling_factor, float):
            latent = latent * self.scaling_factor
        elif isinstance(self.scaling_factor, np.ndarray) and self.latent_channels <= self.scaling_factor.shape[0]:
            latent = (
                latent
                * torch.tensor(self.scaling_factor, dtype=latent.dtype, device=latent.device)[
                    (None, slice(self.latent_channels)) + (None,) * (latent.ndim - 2)
                ]
            )
        else:
            raise ValueError(f"scaling_factor {self.scaling_factor} is not supported")
        return latent

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.model.encode(x)
        return self.apply_scaling_and_shifting_after_encode(latent)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        assert latent.shape[1] == self.latent_channels

        if isinstance(self.scaling_factor, float):
            latent = latent / self.scaling_factor
        elif isinstance(self.scaling_factor, np.ndarray) and self.latent_channels <= self.scaling_factor.shape[0]:
            latent = (
                latent
                / torch.tensor(self.scaling_factor, dtype=latent.dtype, device=latent.device)[
                    (None, slice(self.latent_channels)) + (None,) * (latent.ndim - 2)
                ]
            )
        else:
            raise ValueError(f"scaling_factor {self.scaling_factor} is not supported")

        if self.shifting_factor is None:
            pass
        elif isinstance(self.shifting_factor, float):
            latent = latent - self.shifting_factor
        elif isinstance(self.shifting_factor, np.ndarray) and self.latent_channels <= self.shifting_factor.shape[0]:
            latent = (
                latent
                - torch.tensor(self.shifting_factor, dtype=latent.dtype, device=latent.device)[
                    (None, slice(self.latent_channels)) + (None,) * (latent.ndim - 2)
                ]
            )
        else:
            raise ValueError(f"shifting_factor {self.shifting_factor} is not supported")

        y = self.model.decode(latent)
        return y

    def reconstruct_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        y, info = self.model.reconstruct_image(x)
        info["latent"] = self.apply_scaling_and_shifting_after_encode(info["latent"])
        return y, info

    def reconstruct_video(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        y, info = self.model.reconstruct_video(x)
        info["latent"] = self.apply_scaling_and_shifting_after_encode(info["latent"])
        return y, info


@dataclass
class AutoencoderConfig:
    num_settings: int = 1
    name: Any = None
    scaling_factor: Any = None
    shifting_factor: Any = None
    latent_channels: Any = None


class Autoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        name_list = val2list(cfg.name, cfg.num_settings)
        assert len(name_list) == cfg.num_settings, f"name {cfg.name} is not valid"
        scaling_factor_list = val2list(cfg.scaling_factor, cfg.num_settings)
        assert len(scaling_factor_list) == cfg.num_settings, f"scaling_factor {cfg.scaling_factor} is not valid"
        shifting_factor_list = val2list(cfg.shifting_factor, cfg.num_settings)
        assert len(shifting_factor_list) == cfg.num_settings, f"shifting_factor {cfg.shifting_factor} is not valid"
        latent_channels_list = val2list(cfg.latent_channels, cfg.num_settings)
        assert len(latent_channels_list) == cfg.num_settings, f"latent_channels {cfg.latent_channels} is not valid"

        for i in range(len(scaling_factor_list)):
            if isinstance(scaling_factor_list[i], str):
                scaling_factor_list[i] = np.load(scaling_factor_list[i])

        for i in range(len(shifting_factor_list)):
            if isinstance(shifting_factor_list[i], str):
                shifting_factor_list[i] = np.load(shifting_factor_list[i])

        model_list: list[SingleAutoencoder] = []
        for name, scaling_factor, shifting_factor, latent_channels in zip(
            name_list, scaling_factor_list, shifting_factor_list, latent_channels_list
        ):
            model_list.append(SingleAutoencoder(name, scaling_factor, shifting_factor, latent_channels))
        self.model_list: list[SingleAutoencoder] = nn.ModuleList(model_list)

        self.spatial_compression_ratio: int = self.model_list[0].spatial_compression_ratio
        self.temporal_compression_ratio: int = self.model_list[0].temporal_compression_ratio
        self.temporal_remainder: int = self.model_list[0].temporal_remainder
        assert all(model.spatial_compression_ratio == self.spatial_compression_ratio for model in self.model_list[1:])
        assert all(model.temporal_compression_ratio == self.temporal_compression_ratio for model in self.model_list[1:])
        assert all(model.temporal_remainder == self.temporal_remainder for model in self.model_list[1:])

    def encode(self, x: torch.Tensor, setting_index: int = 0) -> torch.Tensor:
        return self.model_list[setting_index].encode(x)

    def decode(self, latent: torch.Tensor, setting_index: int = 0) -> torch.Tensor:
        return self.model_list[setting_index].decode(latent)

    def reconstruct_image(self, x: torch.Tensor, setting_index: int = 0) -> tuple[torch.Tensor, dict]:
        return self.model_list[setting_index].reconstruct_image(x)

    def reconstruct_video(self, x: torch.Tensor, setting_index: int = 0) -> tuple[torch.Tensor, dict]:
        return self.model_list[setting_index].reconstruct_video(x)

    def forward(self, func_name: str, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)
