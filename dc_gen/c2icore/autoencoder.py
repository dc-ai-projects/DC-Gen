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
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch import nn

from ..ae_model_zoo import (
    DCAE_HF,
    REGISTERED_DCAE_DIFFUSERS_MODEL,
    REGISTERED_DCAE_MODEL,
    REGISTERED_SD_VAE_MODEL,
)
from ..aecore.models.dc_ae import DCAE
from ..aecore.models.sd_vae import SDVAE
from ..models.utils.list import val2list


class SingleAutoencoder(nn.Module):
    model_dict: dict[str, tuple[nn.Module, int, int, Optional[float | np.ndarray], Optional[float | np.ndarray]]] = {}

    @classmethod
    def build_model(cls, model_name: str):
        if model_name in REGISTERED_DCAE_MODEL:
            if REGISTERED_DCAE_MODEL[model_name][1] is None:
                model = DCAE_HF.from_pretrained(f"{REGISTERED_DCAE_MODEL[model_name][2]}/{model_name}")
            else:
                dc_ae_cfg = REGISTERED_DCAE_MODEL[model_name][0](model_name, REGISTERED_DCAE_MODEL[model_name][1])
                model = DCAE(dc_ae_cfg)
            spatial_compression_ratio = model.spatial_compression_ratio
            latent_channels = model.cfg.latent_channels
            scaling_factor = None
            shifting_factor = None
        elif model_name in REGISTERED_DCAE_DIFFUSERS_MODEL:
            from diffusers import AutoencoderDC

            model: AutoencoderDC = AutoencoderDC.from_pretrained(f"mit-han-lab/{model_name}")
            spatial_compression_ratio = model.spatial_compression_ratio
            latent_channels = model.config.latent_channels
            scaling_factor = model.config.scaling_factor
            shifting_factor = None
        elif model_name in REGISTERED_SD_VAE_MODEL:
            model: SDVAE = REGISTERED_SD_VAE_MODEL[model_name][0](model_name, REGISTERED_SD_VAE_MODEL[model_name][1])
            spatial_compression_ratio = model.spatial_compression_ratio
            latent_channels = model.cfg.latent_channels
            if model_name == "vavae-imagenet256-f16d32-dinov2":
                latent_stats = torch.load(
                    "assets/checkpoints/ae/vavae-imagenet256-f16d32-dinov2-latents-stats.pt", weights_only=True
                )
                mean, std = latent_stats["mean"].view(-1), latent_stats["std"].view(-1)  # (x - mean) / std
                shifting_factor = -mean.numpy()
                scaling_factor = 1 / std.numpy()
            else:
                scaling_factor = None
                shifting_factor = None
        elif model_name in [
            "stabilityai/sd-vae-ft-ema",
            "stabilityai/sdxl-vae",
            "flux-vae",
            "sd3-vae",
            "asymmetric-autoencoder-kl-x-1-5",
            "asymmetric-autoencoder-kl-x-2",
        ]:
            import diffusers

            if model_name in ["stabilityai/sd-vae-ft-ema", "stabilityai/sdxl-vae"]:
                model = diffusers.models.AutoencoderKL.from_pretrained(model_name)
                shifting_factor = None
            elif model_name == "flux-vae":
                from diffusers import FluxPipeline

                pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
                model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
                shifting_factor = -model.config.shift_factor
            elif model_name == "sd3-vae":
                from diffusers import StableDiffusion3Pipeline

                pipe = StableDiffusion3Pipeline.from_pretrained(
                    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
                )
                model = diffusers.models.AutoencoderKL.from_pretrained(pipe.vae.config._name_or_path)
                shifting_factor = -model.config.shift_factor
            elif model_name in ["asymmetric-autoencoder-kl-x-1-5", "asymmetric-autoencoder-kl-x-2"]:
                model = diffusers.models.AsymmetricAutoencoderKL.from_pretrained(f"cross-attention/{model_name}")
                shifting_factor = None
            else:
                raise ValueError(f"autoencoder {model_name} is not supported")
            spatial_compression_ratio = 8
            latent_channels = model.config.latent_channels
            scaling_factor = model.config.scaling_factor
        else:
            raise ValueError(f"autoencoder {model_name} is not supported")

        cls.model_dict[model_name] = (
            model.eval(),
            spatial_compression_ratio,
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_name in REGISTERED_DCAE_MODEL:
            if self.latent_channels != self.default_latent_channels:
                latent = self.model.encode(x, latent_channels=self.latent_channels)
            else:
                latent = self.model.encode(x)
        elif self.model_name in REGISTERED_SD_VAE_MODEL:
            latent = self.model.encode(x)
        elif self.model_name in REGISTERED_DCAE_DIFFUSERS_MODEL:
            latent = self.model.encode(x).latent
        elif self.model_name in [
            "stabilityai/sd-vae-ft-ema",
            "stabilityai/sdxl-vae",
            "flux-vae",
            "sd3-vae",
            "asymmetric-autoencoder-kl-x-1-5",
            "asymmetric-autoencoder-kl-x-2",
        ]:
            latent = self.model.encode(x).latent_dist.sample()
        else:
            raise ValueError(f"autoencoder {self.model_name} is not supported")
        assert latent.shape[1] == self.latent_channels

        if self.shifting_factor is None:
            pass
        elif isinstance(self.shifting_factor, float):
            latent = latent + self.shifting_factor
        elif isinstance(self.shifting_factor, np.ndarray) and self.latent_channels <= self.shifting_factor.shape[0]:
            latent = (
                latent
                + torch.tensor(self.shifting_factor, dtype=latent.dtype, device=latent.device)[
                    None, : self.latent_channels, None, None
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
                    None, : self.latent_channels, None, None
                ]
            )
        else:
            raise ValueError(f"scaling_factor {self.scaling_factor} is not supported")
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        assert latent.shape[1] == self.latent_channels

        if isinstance(self.scaling_factor, float):
            latent = latent / self.scaling_factor
        elif isinstance(self.scaling_factor, np.ndarray) and self.latent_channels <= self.scaling_factor.shape[0]:
            latent = (
                latent
                / torch.tensor(self.scaling_factor, dtype=latent.dtype, device=latent.device)[
                    None, : self.latent_channels, None, None
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
                    None, : self.latent_channels, None, None
                ]
            )
        else:
            raise ValueError(f"shifting_factor {self.shifting_factor} is not supported")

        if self.model_name in REGISTERED_DCAE_MODEL or self.model_name in REGISTERED_SD_VAE_MODEL:
            y = self.model.decode(latent)
        elif self.model_name in REGISTERED_DCAE_DIFFUSERS_MODEL:
            y = self.model.decode(latent, return_dict=False)[0]
        elif self.model_name in [
            "stabilityai/sd-vae-ft-ema",
            "stabilityai/sdxl-vae",
            "flux-vae",
            "sd3-vae",
            "asymmetric-autoencoder-kl-x-1-5",
            "asymmetric-autoencoder-kl-x-2",
        ]:
            y = self.model.decode(latent).sample
        else:
            raise ValueError(f"autoencoder {self.model_name} is not supported")
        return y


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
        self.model_list: Mapping[int, SingleAutoencoder] | nn.ModuleList = nn.ModuleList(model_list)

        self.spatial_compression_ratio = self.model_list[0].spatial_compression_ratio
        assert all(model.spatial_compression_ratio == self.spatial_compression_ratio for model in self.model_list[1:])

    def encode(self, x: torch.Tensor, setting_index: int = 0) -> torch.Tensor:
        return self.model_list[setting_index].encode(x)

    def decode(self, latent: torch.Tensor, setting_index: int = 0) -> torch.Tensor:
        return self.model_list[setting_index].decode(latent)
