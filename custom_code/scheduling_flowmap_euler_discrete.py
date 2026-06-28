# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FlowMapDiscreteScheduler(SchedulerMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.set_timesteps(num_train_timesteps, device='cpu')
        self.set_train_weight()

    def adaptive_weighting(self, loss, p=1.0, eps=1e-3):
        weight = 1.0 / torch.pow(loss.detach() + eps, p)
        return weight * loss

    def set_train_weight(self):
        x = self.timesteps
        y = torch.exp(-2 * ((x - self.config.num_train_timesteps / 2) / self.config.num_train_timesteps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (self.config.num_train_timesteps / y_shifted.sum())
        self.linear_timesteps_weights = bsmntw_weighing

    @torch.no_grad()
    def get_train_weight(self, timesteps):
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(self.timesteps.device)).abs(), dim=0).reshape(timesteps.shape)  # noqa: E501
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timesteps.device)

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        sample = timestep * noise + (1.0 - timestep) * sample
        return sample

    def apply_shift(self, sigmas):
        """Apply shift transformation to sigmas/timesteps."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)
        timesteps = self.apply_shift(timesteps)

        self.timesteps = timesteps * self.config.num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        sample: torch.FloatTensor,
        timestep: Optional[Union[float, torch.FloatTensor]] = None,
        r_timestep: Optional[Union[float, torch.FloatTensor]] = None,
    ):
        timestep = timestep / self.config.num_train_timesteps
        r_timestep = r_timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output
        return prev_sample.to(model_output.dtype)
