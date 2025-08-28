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

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from omegaconf import MISSING

from ....models.utils.network import get_device
from ...models.base import BaseC2IModel, BaseC2IModelConfig
from .sit_sampler import create_transport as sit_create_transport
from .sit_sampler.transport import Sampler as SiTSampler
from .uni_sampler import UniSampler, UniSamplerConfig, UniTrainScheduler, UniTrainSchedulerConfig


@dataclass
class BaseDiffusionModelConfig(BaseC2IModelConfig):
    cfg_channels: Optional[int] = None
    adaptive_channel: bool = False
    adaptive_channel_share_weights: bool = True

    train_scheduler: str = MISSING
    eval_scheduler: str = MISSING
    num_inference_steps: int = MISSING

    flow_shift: float = 3.0

    reverse_time: bool = False  # if sample process starts with 0 and ends with T>0

    cfg_schedule: str = "constant"
    cfg_schedule_cosine_pow: float = 1.0
    use_guidance_interval: bool = False
    guidance_t_min: float = 0.2
    guidance_t_max: float = 0.8

    uni_sampler: UniSamplerConfig = field(default_factory=lambda: UniSamplerConfig(reverse_time="${..reverse_time}"))
    count_nfe: bool = False  # count number of function evaluations

    uni_train_scheduler: UniTrainSchedulerConfig = field(default_factory=UniTrainSchedulerConfig)


class BaseDiffusionModel(BaseC2IModel):
    def __init__(self, cfg: BaseDiffusionModelConfig):
        super().__init__(cfg)
        self.cfg: BaseDiffusionModelConfig

        # scheduler
        if cfg.train_scheduler == "GaussianDiffusion":
            from .dit_sampler import create_diffusion

            self.train_scheduler = create_diffusion(timestep_respacing="")
        elif cfg.train_scheduler == "DPM_Solver":
            _betas = (torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2).numpy()
            from .uvit_sampler.uvit_schedule import UViTSchedule

            self.train_scheduler = UViTSchedule(_betas)
        elif cfg.train_scheduler == "SiTSampler":
            from .sit_sampler import create_transport

            self.transport = create_transport("Linear", "velocity", None, None, None)
        elif cfg.train_scheduler == "SanaScheduler":
            from .sana_utils.sana_sampler import SanaTrainScheduler

            self.train_scheduler = SanaTrainScheduler(
                str(1000),
                noise_schedule="linear_flow",
                predict_v=True,
                learn_sigma=False,
                pred_sigma=False,
                snr=False,
                flow_shift=cfg.flow_shift,
            )
        elif cfg.train_scheduler == "UniTrainScheduler":
            self.train_scheduler = UniTrainScheduler(cfg.uni_train_scheduler)
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

        if cfg.eval_scheduler == "GaussianDiffusion":
            from .dit_sampler import create_diffusion

            self.eval_scheduler = create_diffusion(str(250))
        elif cfg.eval_scheduler == "UniPC":
            from diffusers import UniPCMultistepScheduler

            self.eval_scheduler = UniPCMultistepScheduler(
                solver_order=3,
                # rescale_betas_zero_snr=False,
                prediction_type="epsilon",
            )
        elif cfg.eval_scheduler == "DPMSolverSinglestep":
            from diffusers import DPMSolverSinglestepScheduler

            self.eval_scheduler = DPMSolverSinglestepScheduler()
        elif cfg.eval_scheduler == "DPM_Solver":
            from .uvit_sampler.dpm_solver_pp import NoiseScheduleVP

            device = torch.device("cuda")
            _betas = (torch.linspace(0.00085**0.5, 0.0120**0.5, 1000, dtype=torch.float64) ** 2).numpy()
            self.eval_scheduler = NoiseScheduleVP(
                schedule="discrete", betas=torch.tensor(_betas, device=device).float()
            )
        elif cfg.eval_scheduler in ["ODE_dopri5", "ODE_heun2", "SDE"]:
            # assert cfg.train_scheduler == "SiTSampler"
            transport = sit_create_transport("Linear", "velocity", None, None, None)
            sampler = SiTSampler(transport)
            if cfg.eval_scheduler == "ODE_dopri5":
                self.eval_scheduler = sampler.sample_ode(
                    sampling_method="dopri5", num_steps=cfg.num_inference_steps, atol=1e-6, rtol=0.001, reverse=False
                )
            elif cfg.eval_scheduler == "ODE_heun2":
                self.eval_scheduler = sampler.sample_ode(
                    sampling_method="heun2", num_steps=cfg.num_inference_steps, atol=1e-6, rtol=0.001, reverse=False
                )
            elif cfg.eval_scheduler == "SDE":
                self.eval_scheduler = sampler.sample_sde(
                    sampling_method="Euler",
                    diffusion_form="sigma",
                    diffusion_norm=1.0,
                    last_step="Mean",
                    last_step_size=0.04,
                    num_steps=cfg.num_inference_steps,
                )
            else:
                raise ValueError(f"eval scheduler {cfg.eval_scheduler} is not supported")
        elif cfg.eval_scheduler == "SanaScheduler":
            from diffusers import DPMSolverMultistepScheduler

            self.eval_scheduler = DPMSolverMultistepScheduler(
                flow_shift=cfg.flow_shift, use_flow_sigmas=True, prediction_type="flow_prediction"
            )
        elif cfg.eval_scheduler == "UniSampler":
            self.eval_scheduler = UniSampler(cfg.uni_sampler)
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

    def enable_activation_checkpointing(self, mode: str):
        raise NotImplementedError

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward_without_cfg(combined, t, y)[0]
        if self.cfg.cfg_channels is not None:
            eps, rest = model_out[:, : self.cfg.cfg_channels], model_out[:, self.cfg.cfg_channels :]
        else:
            eps, rest = model_out[:, : self.cfg.in_channels], model_out[:, self.cfg.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        if self.cfg.use_guidance_interval and (
            t[0].item() < self.cfg.guidance_t_min or t[0].item() > self.cfg.guidance_t_max
        ):
            cfg_scale = 1.0
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor,
        null_inputs: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        channels: Optional[int] = None,
    ) -> torch.Tensor:
        device = get_device(self)
        if channels is None:
            channels = self.cfg.in_channels
        samples = torch.randn(
            (inputs.shape[0], channels, self.cfg.input_size, self.cfg.input_size),
            generator=generator,
            device=device,
        )

        if cfg_scale != 1.0:
            assert null_inputs is not None
            samples = torch.cat([samples, samples], dim=0)
            inputs = torch.cat([inputs, null_inputs], dim=0)
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs, cfg_scale=cfg_scale)
                samples = self.eval_scheduler.p_sample_loop(
                    self.forward_with_cfg,
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                )
            elif self.cfg.eval_scheduler in ["UniPC", "DPMSolverSinglestep", "SanaScheduler"]:
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_with_cfg(samples, timesteps, inputs, cfg_scale)
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            elif self.cfg.eval_scheduler == "DPM_Solver":
                from .uvit_sampler.dpm_solver_pp import DPM_Solver

                N = self.eval_scheduler.total_N
                dpm_solver = DPM_Solver(
                    lambda x, t_continuous: self.forward_with_cfg(x, t_continuous * N, inputs, cfg_scale),
                    self.eval_scheduler,
                    predict_x0=True,
                    thresholding=False,
                )
                samples = dpm_solver.sample(samples, steps=self.cfg.num_inference_steps, eps=1.0 / N, T=1.0)
            elif self.cfg.eval_scheduler in ["ODE_dopri5", "ODE_heun2", "SDE"]:
                samples = self.eval_scheduler(samples, self.forward_with_cfg, y=inputs, cfg_scale=cfg_scale)[-1]
            elif self.cfg.eval_scheduler == "UniSampler":
                self.eval_scheduler.set_model_fn(
                    lambda x, t_continuous: self.forward_with_cfg(x, t_continuous, inputs, cfg_scale)
                )
                samples = self.eval_scheduler.sample(samples, self.cfg.num_inference_steps, generator=generator)
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")
            samples, _ = samples.chunk(2, dim=0)
        else:
            if self.cfg.eval_scheduler == "GaussianDiffusion":
                model_kwargs = dict(y=inputs)
                samples = self.eval_scheduler.p_sample_loop(
                    lambda x, t, y: self.forward_without_cfg(x, t, y)[0],
                    samples.shape,
                    samples,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    device=device,
                )
            elif self.cfg.eval_scheduler in ["UniPC", "DPMSolverSinglestep", "SanaScheduler"]:
                self.eval_scheduler.set_timesteps(num_inference_steps=self.cfg.num_inference_steps)
                for t in self.eval_scheduler.timesteps:
                    timesteps = torch.tensor([t] * samples.shape[0], device=device).int()
                    model_output = self.forward_without_cfg(samples, timesteps, inputs)[0]
                    samples = self.eval_scheduler.step(model_output, t, samples).prev_sample
            elif self.cfg.eval_scheduler == "DPM_Solver":
                from .uvit_sampler.dpm_solver_pp import DPM_Solver

                N = self.eval_scheduler.total_N
                dpm_solver = DPM_Solver(
                    lambda x, t_continuous: self.forward_without_cfg(x, t_continuous * N, inputs)[0],
                    self.eval_scheduler,
                    predict_x0=True,
                    thresholding=False,
                )
                samples = dpm_solver.sample(samples, steps=self.cfg.num_inference_steps, eps=1.0 / N, T=1.0)
            elif self.cfg.eval_scheduler in ["ODE_dopri5", "ODE_heun2", "SDE"]:
                samples = self.eval_scheduler(samples, lambda x, t, y: self.forward_without_cfg(x, t, y)[0], y=inputs)[
                    -1
                ]
            elif self.cfg.eval_scheduler == "UniSampler":
                self.eval_scheduler.set_model_fn(
                    lambda x, t_continuous: self.forward_without_cfg(x, t_continuous, inputs)[0]
                )
                samples = self.eval_scheduler.sample(samples, self.cfg.num_inference_steps, generator=generator)
            else:
                raise NotImplementedError(f"eval scheduler {self.cfg.eval_scheduler} is not supported")

        return samples

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        projections: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        info = {}
        detailed_loss_dict = {}
        device = x.device
        if self.cfg.train_scheduler == "GaussianDiffusion":
            model_kwargs = dict(y=y)
            timesteps = torch.randint(0, self.train_scheduler.num_timesteps, (x.shape[0],), device=device)
            scheduler_output = self.train_scheduler.training_losses(
                lambda x, t, y: self.forward_without_cfg(x, t, y)[0], x, timesteps, model_kwargs
            )
            loss = scheduler_output["loss"].mean()
        elif self.cfg.train_scheduler == "DPM_Solver":
            n, eps, xn = self.train_scheduler.sample(x)  # n in {1, ..., 1000}
            eps_pred = self.forward_without_cfg(xn, n, y)[0]
            loss = (eps - eps_pred).square().mean()
        elif self.cfg.train_scheduler == "SiTSampler":
            model_kwargs = dict(y=y)
            scheduler_output = self.transport.training_losses(
                lambda x, t, y: self.forward_without_cfg(x, t, y)[0], x, model_kwargs
            )
            loss = scheduler_output["loss"].mean()
        elif self.cfg.train_scheduler == "SanaScheduler":
            from .sana_utils.sana_sampler import compute_density_for_timestep_sampling

            bs = x.shape[0]
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bs,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=None,  # not used
            )
            timesteps = (u * 1000).long().to(x.device)
            model_kwargs = dict(y=y)
            scheduler_output = self.train_scheduler.training_losses(
                lambda x, t, y: self.forward_without_cfg(x, t, y)[0], x, timesteps, model_kwargs
            )
            loss = scheduler_output["loss"].mean()
        elif self.cfg.train_scheduler == "UniTrainScheduler":
            t, x_t, model_target = self.train_scheduler.schedule(x, generator)
            model_output = self.forward_without_cfg(x_t, t, y)[0]
            loss = (model_output - model_target).square().mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")
        detailed_loss_dict["loss"] = loss
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
