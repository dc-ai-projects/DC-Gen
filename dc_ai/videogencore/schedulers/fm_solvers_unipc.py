# Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# Convert unipc for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)
from diffusers.utils import deprecate


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: Optional[float] = 1.0,
    ):

        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()

        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * 2
        self.timestep_list = [None] * 2
        self.lower_order_nums = 0
        self.last_sample = None
        self._step_index = None
        self._begin_index = None

        self.sigmas = self.sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: Union[int, None] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        shift: Optional[Union[float, None]] = None,
    ):
        if sigmas is None:
            sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]  # pyright: ignore

        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        sigma_last = 0

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)  # pyright: ignore

        self.sigmas = torch.from_numpy(sigmas)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.num_inference_steps = len(timesteps)

        self.model_outputs = [None] * 2
        self.lower_order_nums = 0
        self.last_sample = None

        self._step_index = None
        self._begin_index = None
        self.sigmas = self.sigmas.to("cpu")

    def _sigma_to_alpha_sigma_t(self, sigma):
        return 1 - sigma, sigma

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        *args,
        sample: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        sigma = self.sigmas[self._step_index]
        _, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

        sigma_t = self.sigmas[self._step_index]
        x0_pred = sample - sigma_t * model_output

        return x0_pred

    def multistep_uni_p_bh_update(
        self,
        *args,
        sample: torch.Tensor = None,
        order: int = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = sample

        sigma_t, sigma_s0 = self.sigmas[self._step_index + 1], self.sigmas[self._step_index]  # pyright: ignore
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - i  # pyright: ignore
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)  # pyright: ignore

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        B_h = torch.expm1(hh)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            D1s = None

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0

        if D1s is not None:
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)  # pyright: ignore
        else:
            pred_res = 0

        x_t = x_t_ - alpha_t * B_h * pred_res
        x_t = x_t.to(x.dtype)

        return x_t

    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        *args,
        last_sample: torch.Tensor = None,
        this_sample: torch.Tensor = None,
        order: int = None,  # pyright: ignore
        **kwargs,
    ) -> torch.Tensor:
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[self._step_index], self.sigmas[self._step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0
        device = this_sample.device

        rks = []
        D1s = []
        for i in range(1, order):
            si = self._step_index - (i + 1)  # pyright: ignore
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)  # pyright: ignore

        rks.append(1.0)
        rks = torch.tensor(rks, device=device)

        R = []
        b = []

        hh = -h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        B_h = torch.expm1(hh)

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=device)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        if order == 1:
            rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0

        if D1s is not None:
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        else:
            corr_res = 0

        D1_t = model_t - m0
        x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)

        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep[0:1]).nonzero()

        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep):
        if self._begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self, model_output: torch.Tensor, timestep: Union[int, torch.Tensor], sample: torch.Tensor
    ) -> Union[SchedulerOutput, Tuple]:

        if self._step_index is None:
            self._init_step_index(timestep)

        use_corrector = self._step_index > 0 and self.last_sample is not None  # pyright: ignore

        model_output_convert = self.convert_model_output(model_output, sample=sample)
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
            )

        self.model_outputs[0] = self.model_outputs[1]
        self.timestep_list[0] = self.timestep_list[1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep  # pyright: ignore

        this_order = min(2, len(self.timesteps) - self._step_index)  # pyright: ignore

        self.this_order = min(this_order, self.lower_order_nums + 1)  # warmup for multistep
        assert self.this_order > 0

        self.last_sample = sample

        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,  # pass the original non-converted model output, in case solver-p is used
            sample=sample,
            order=self.this_order,
        )

        if self.lower_order_nums < 2:
            self.lower_order_nums += 1

        self._step_index += 1  # pyright: ignore

        return prev_sample
