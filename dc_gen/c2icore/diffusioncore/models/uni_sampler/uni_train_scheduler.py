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
from typing import Optional, Tuple

import torch

from ..dit_sampler import gaussian_diffusion as gd
from .noise_schedule import NoiseScheduleCosine, NoiseScheduleLinear, NoiseScheduleVP


@dataclass
class UniTrainSchedulerConfig:
    model_prediction_type: str = "velocity"  # model prediction type during training
    noise_schedule: str = "linear"  # noise schedule type
    time_input_type: str = "continuous"  # time input type ["continuous", "discrete_999", "discrete_1000"]
    reverse_time: bool = False  # reverse time, if x_t ~ clean data when t=t_max
    time_schedule: str = "uniform_t"  # training time schedule type
    logit_mean: float = 0.0
    logit_std: float = 1.0
    # warning: t_min should >= 1e-3 when using discrete time schedule
    t_min: float = 1e-5  # training range
    t_max: float = 1  # training range


class UniTrainScheduler:
    """
    1. Denoising loss: support ['noise', 'data', 'velocity', 'score'] as prediction type
    2. Noise schedule: support ['linear_flow', 'cosine', 'vp_gaussian', 'vp_dpmsolver'] as noise schedule
    3. Training time schedule: support ['uniform_t', 'uniform_logSNR', 'logit_normal'] TODO support dynamic time schedule
    """

    def __init__(self, cfg: UniTrainSchedulerConfig):
        self.cfg = cfg
        if cfg.noise_schedule == "linear_flow":
            self.noise_schedule = NoiseScheduleLinear()
        elif cfg.noise_schedule == "cosine":
            self.noise_schedule = NoiseScheduleCosine()
        elif cfg.noise_schedule == "vp_dpmsolver":
            device = torch.device("cuda")
            _betas = (torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2).numpy()
            self.noise_schedule = NoiseScheduleVP(
                schedule="discrete", betas=torch.tensor(_betas, device=device).float()
            )
        elif cfg.noise_schedule == "vp_gaussian":
            device = torch.device("cuda")
            _betas = gd.get_named_beta_schedule("linear", 1000)
            self.noise_schedule = NoiseScheduleVP(
                schedule="discrete", betas=torch.tensor(_betas, device=device).float()
            )
        else:
            raise NotImplementedError(f"noise_schedule {cfg.noise_schedule} not implemented")

        if "discrete" in self.cfg.time_input_type:
            assert self.cfg.t_min >= 1e-3, "t_min should >= 1e-3 when using discrete time schedule"

        if self.cfg.reverse_time:
            assert self.cfg.time_input_type == "continuous", "reverse_time only support continuous time schedule"

    def get_velocity_from_noise_and_data(self, t: torch.Tensor, noise: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Get velocity from x (pure data) and noise
        """
        d_alpha_t = self.noise_schedule.marginal_d_alpha(t).view(-1, *([1] * len(x.shape[1:])))
        d_sigma_t = self.noise_schedule.marginal_d_std(t).view(-1, *([1] * len(x.shape[1:])))

        velocity = d_alpha_t * x + d_sigma_t * noise

        return velocity

    def get_random_continuous_time(
        self, batch_size: int, device: torch.device, generator: Optional[torch.Generator]
    ) -> torch.Tensor:
        """
        Get continuous time from predefined distribution. t ~ [t_min, t_max]
        t.shpe = [batch_size], batch_size = x.size(0)
        """
        if self.cfg.time_schedule == "uniform_t":
            u = torch.rand(batch_size, device=device, generator=generator)  # [batch_size]
            t = u * (self.cfg.t_max - self.cfg.t_min) + self.cfg.t_min
            t = t.to("cuda")
        elif self.cfg.time_schedule == "uniform_logSNR":
            lambda_min = self.noise_schedule.marginal_lambda(self.cfg.t_max)
            lambda_max = self.noise_schedule.marginal_lambda(self.cfg.t_min)
            u = torch.rand(batch_size, device=device, generator=generator)  # [batch_size]
            lambda_t = u * (lambda_max - lambda_min) + lambda_min
            lambda_t = lambda_t.to("cuda")
            t = self.noise_schedule.inverse_lambda(lambda_t)
        elif self.cfg.time_schedule == "logit_normal":
            mean = torch.tensor(self.cfg.logit_mean, device="cuda")
            std = torch.tensor(self.cfg.logit_std, device="cuda")
            t = torch.normal(mean=mean.expand(batch_size), std=std.expand(batch_size), generator=generator)
            t = torch.sigmoid(t)
        else:
            raise NotImplementedError(f"Time schedule {self.cfg.time_schedule} not implemented")

        if "discrete" in self.cfg.time_input_type:
            # here we set t_i = (i + 1) / N following DPM-Solver, which means t ~ [1e-3, 1]
            # if model input type is discrete_999, t_discrete will be converted to {0, 1, 2, ..., 999} in the end
            t = torch.round(t * 1000) / 1000
        elif self.cfg.time_input_type == "continuous":
            pass
        else:
            raise NotImplementedError(f"time_input_type {self.cfg.time_input_type} not implemented")
        return t

    def schedule(
        self, x: torch.Tensor, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        input: x: clean data tensor [batch_size, ...], generator: torch.Generator
        output: t: time tensor [batch_size], x_t: noisy data tensor [batch_size, ...], model_target: target tensor [batch_size, ...]
        """
        batch_size = x.size(0)
        t = self.get_random_continuous_time(batch_size, x.device, generator)
        alpha_t = self.noise_schedule.marginal_alpha(t)  # [batch_size]
        sigma_t = self.noise_schedule.marginal_std(t)  # [batch_size]
        # expand alpha_t and sigma_t to [batch_size, ...]
        alpha_t = alpha_t.view(batch_size, *([1] * len(x.shape[1:])))
        sigma_t = sigma_t.view(batch_size, *([1] * len(x.shape[1:])))
        noise = torch.randn(x.size(), generator=generator, device=x.device, dtype=x.dtype)
        x_t = alpha_t * x + sigma_t * noise

        if self.cfg.model_prediction_type == "noise":
            model_target = noise
        elif self.cfg.model_prediction_type == "data":
            model_target = x
        elif self.cfg.model_prediction_type == "velocity":
            model_target = self.get_velocity_from_noise_and_data(t, noise, x)
        elif self.cfg.model_prediction_type == "score":
            model_target = -noise / sigma_t
        else:
            raise NotImplementedError(f"model prediction type {self.cfg.model_prediction_type} not implemented")

        # convert the time to time_input_type
        # when using discrete time schedule, t_continuous ~ [1e-3, 1]
        # t_discrete will be converted to {0, 1, ..., 999} or {1, 2, ..., 1000}
        if self.cfg.time_input_type == "discrete_999":
            # t_discrete ~ {0, 1, 2, ..., 999}
            t = torch.round(t * 1000) - 1
        elif self.cfg.time_input_type == "discrete_1000":
            # t_discrete ~ {1, 2, ..., 1000}
            t = torch.round(t * 1000)
        elif self.cfg.time_input_type == "continuous":
            if self.cfg.reverse_time:
                t = 1 - t
                if self.cfg.model_prediction_type == "velocity":
                    model_target = -model_target
        else:
            raise NotImplementedError(f"time_input_type {self.cfg.time_input_type} not implemented")

        return t, x_t, model_target
