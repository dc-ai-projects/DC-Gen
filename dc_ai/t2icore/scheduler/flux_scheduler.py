# Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
# The original implementation is by Stability AI, Katherine Crowson and The HuggingFace Team, licensed under the Apache License 2.0.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class FlowMatchEulerDiscreteSchedulerConfig:
    num_train_timesteps: int = 1000
    shift: float = 1.0
    use_dynamic_shifting: bool = False
    base_shift: Optional[float] = 0.5
    max_shift: Optional[float] = 1.15
    base_image_seq_len: Optional[int] = 256
    max_image_seq_len: Optional[int] = 4096


class FlowMatchEulerDiscreteScheduler:
    def __init__(self, cfg: FlowMatchEulerDiscreteSchedulerConfig):
        self.cfg = cfg

        timesteps = np.linspace(1, self.cfg.num_train_timesteps, self.cfg.num_train_timesteps, dtype=np.float32)[
            ::-1
        ].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / self.cfg.num_train_timesteps

        self.timesteps = sigmas * self.cfg.num_train_timesteps

        self.step_index = None
        self.begin_index = None

        self.shift = self.cfg.shift

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        sigmas = np.array(sigmas).astype(np.float32)
        num_inference_steps = len(sigmas)
        self.num_inference_steps = num_inference_steps

        if self.cfg.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.cfg.shift * sigmas / (1 + (self.cfg.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.cfg.num_train_timesteps

        sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self.timesteps = timesteps.to(device=device)
        self.sigmas = sigmas
        self.step_index = None
        self.begin_index = None

    def index_for_timestep(self, timestep):
        indices = (self.timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def init_step_index(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        self.step_index = self.index_for_timestep(timestep)

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> torch.FloatTensor:

        self.init_step_index(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        self.step_index += 1

        return prev_sample


class FluxScheduler:
    def __init__(self, num_inference_steps: int = 20, shift: float = 3.0, use_dynamic_shifting: bool = True):
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        self.scheduler_config = FlowMatchEulerDiscreteSchedulerConfig(
            shift=shift, use_dynamic_shifting=use_dynamic_shifting
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler(cfg=self.scheduler_config)

    def calculate_shift(
        self,
        image_seq_len: int = 4096,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
    ):
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

        return timesteps, num_inference_steps

    def get_timesteps(self, device):
        sigmas = np.linspace(1.0, 1 / self.num_inference_steps, self.num_inference_steps)
        if self.scheduler.cfg.use_dynamic_shifting:
            mu = self.calculate_shift(
                self.scheduler.cfg.max_image_seq_len,
                self.scheduler.cfg.base_image_seq_len,
                self.scheduler.cfg.max_image_seq_len,
                self.scheduler.cfg.base_shift,
                self.scheduler.cfg.max_shift,
            )
        else:
            mu = None
        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        return timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> torch.FloatTensor:
        return self.scheduler.step(model_output, timestep, sample, return_dict=return_dict)
