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

from ..dit_sampler import gaussian_diffusion as gd
from .dpm_solver import DPM_Solver
from .model_wrapper import model_wrapper
from .noise_schedule import NoiseScheduleCosine, NoiseScheduleLinear, NoiseScheduleVP
from .sde_solver import SDE_Solver
from .uni_pc import UniPC


@dataclass
class UniSamplerConfig:
    model_prediction_type: str = "velocity"  # original model prediction type
    # prediction type used by the solver
    solver_prediction_type: str = "data"
    # ["linear_flow", "vp_dpmsolver", "vp_gaussian"]
    noise_schedule: str = "linear_flow"
    # ["continuous", "discrete_999", "discrete_1000"]
    time_input_type: str = "continuous"
    # if sample process starts with 0 and ends with T>0
    reverse_time: bool = False
    # ODE_dopri5, ODE_heun2, DPM_Solver  # TODO add UniPC and USF
    solver: str = "DPM_Solver"
    solver_order: int = 1  # order of the ODE solver
    # ["time_uniform", "time_quadratic", "logSNR"]
    skip_type: str = "time_uniform"
    sde_diffusion_coefficient: str = "increasing-decreasing"
    sde_diffusion_norm: float = 1.0
    sde_algorithm_type: str = "Euler"

    t_start: float = 1.0  # might cause numerical instability when t_start > 1.0 - 1e-4
    t_end: float = 1e-3  # might cause numerical instability when t_end < 1e-3


class UniSampler:
    def __init__(self, cfg: UniSamplerConfig):
        self.cfg = cfg
        # TODO support more kinds of noise schedule
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

    def set_model_fn(self, model_fn):
        self.noise_prediction_fn = model_wrapper(
            model_fn,
            self.noise_schedule,
            time_input_type=self.cfg.time_input_type,
            model_type=self.cfg.model_prediction_type,
            reverse_time=self.cfg.reverse_time,
        )
        if self.cfg.solver == "DPM_Solver":
            self.solver = DPM_Solver(
                self.noise_prediction_fn,
                self.noise_schedule,
                algorithm_type="dpmsolver++" if self.cfg.solver_prediction_type == "data" else "dpmsolver",
            )
        elif self.cfg.solver == "UniPC":
            self.solver = UniPC(
                self.noise_prediction_fn,
                self.noise_schedule,
                algorithm_type="data_prediction" if self.cfg.solver_prediction_type == "data" else "noise_prediction",
            )
        elif self.cfg.solver == "SDE_Solver":
            self.solver = SDE_Solver(
                self.noise_prediction_fn,
                self.noise_schedule,
                algorithm_type=self.cfg.sde_algorithm_type,
                diffusion_coefficient_type=self.cfg.sde_diffusion_coefficient,
                diffusion_norm=self.cfg.sde_diffusion_norm,
            )
        else:
            raise NotImplementedError(f"ODE solver {self.cfg.solver} not implemented")

    def sample(self, x: torch.Tensor, steps: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        if self.cfg.solver == "DPM_Solver":
            return self.solver.sample(
                x,
                steps,
                t_start=self.cfg.t_start,
                t_end=self.cfg.t_end,
                order=self.cfg.solver_order,
                skip_type=self.cfg.skip_type,
            )
        elif self.cfg.solver == "UniPC":
            return self.solver.sample(
                x,
                steps,
                t_start=self.cfg.t_start,
                t_end=self.cfg.t_end,
                order=self.cfg.solver_order,
                skip_type=self.cfg.skip_type,
            )
        elif self.cfg.solver == "SDE_Solver":
            return self.solver.sample(
                x,
                steps,
                t_start=self.cfg.t_start,
                t_end=self.cfg.t_end,
                skip_type=self.cfg.skip_type,
                generator=generator,
            )
        else:
            raise NotImplementedError(f"ODE solver {self.cfg.solver} not implemented")
