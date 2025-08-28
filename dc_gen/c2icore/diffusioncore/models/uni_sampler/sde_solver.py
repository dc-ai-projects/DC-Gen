# Modified from SiT's repo: https://github.com/willisma/SiT
# The original implementation is by Meta Platforms, Inc. and affiliates, licensed under the MIT License. See https://github.com/willisma/SiT.

import math
from typing import Optional

import torch

from .noise_schedule import NoiseSchedule


class SDE_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule: NoiseSchedule,
        algorithm_type: str = "Euler",
        diffusion_coefficient_type: str = "constant",
        diffusion_norm=1.0,
    ):
        self.model_fn = lambda x, t: model_fn(x, t.expand((x.shape[0])))  # model_fn is a noise prediction function
        self.noise_schedule = noise_schedule
        self.algorithm_type = algorithm_type
        self.diffusion_coefficient_type = diffusion_coefficient_type
        self.diffusion_norm = diffusion_norm

    def get_score_and_velocity(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        noise = self.model_fn(x, t)
        sigma_t = self.noise_schedule.marginal_std(t)
        alpha_t = self.noise_schedule.marginal_alpha(t)
        score = -noise / sigma_t

        d_alpha_t = self.noise_schedule.marginal_d_alpha(t)
        d_sigma_t = self.noise_schedule.marginal_d_std(t)
        velocity = (d_alpha_t * x - (d_alpha_t * sigma_t - alpha_t * d_sigma_t) * noise) / alpha_t

        return score, velocity

    def sample(
        self,
        x: torch.Tensor,
        steps: int,
        t_start: float,
        t_end: float,
        skip_type: str,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        device = x.device
        time_steps = self.get_time_steps(skip_type, t_T, t_0, steps, device)
        for i in range(steps - 1):
            t = time_steps[i]
            dt = time_steps[i + 1] - t
            x = self.step(x, t, dt, generator)
        # last step
        x = self.__last_step(x, time_steps[-2], time_steps[-1] - time_steps[-2])
        return x

    def step(
        self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        if self.algorithm_type == "Euler":
            return self.__Euler_Maruyama_step(x, t, dt, generator)
        elif self.algorithm_type == "Heun":
            return self.__Heun_step(x, t, dt, generator)
        else:
            raise NotImplementedError(f"Algorithm type {self.algorithm_type} not implemented.")

    def __last_step(self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        diffusion_coefficient = self.get_diffusion_coefficient(t)
        score, velocity = self.get_score_and_velocity(x, t)
        drift = velocity - score * diffusion_coefficient / 2
        x = x + drift * dt
        return x

    def __Euler_Maruyama_step(
        self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        diffusion_coefficient = self.get_diffusion_coefficient(t)
        score, velocity = self.get_score_and_velocity(x, t)
        drift = velocity - score * diffusion_coefficient / 2
        w = torch.randn(x.size(), generator=generator, device=x.device, dtype=x.dtype)
        dw = torch.sqrt(torch.abs(dt)) * w
        x = x + drift * dt + torch.sqrt(diffusion_coefficient) * dw
        return x

    def __Heun_step(
        self, x: torch.Tensor, t: torch.Tensor, dt: torch.Tensor, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        dc1 = self.get_diffusion_coefficient(t)
        score1, velocity1 = self.get_score_and_velocity(x, t)
        drift1 = velocity1 - score1 * dc1 / 2
        w = torch.randn(x.size(), generator=generator, device=x.device, dtype=x.dtype)
        dw = torch.sqrt(torch.abs(dt)) * w

        x_tilde = x + drift1 * dt + torch.sqrt(dc1) * dw

        dc2 = self.get_diffusion_coefficient(t + dt)
        score2, velocity2 = self.get_score_and_velocity(x_tilde, t + dt)
        drift2 = velocity2 - score2 * dc2 / 2

        average_drift = (drift1 + drift2) / 2
        average_diffusion = (torch.sqrt(dc1) + torch.sqrt(dc2)) / 2

        x_new = x + average_drift * dt + average_diffusion * dw
        return x_new

    def get_diffusion_coefficient(self, t: torch.Tensor) -> torch.Tensor:
        choices = {
            "constant": self.diffusion_norm,
            "sigma": self.diffusion_norm * self.noise_schedule.marginal_std(t),
            "linear": self.diffusion_norm * t,
            "decreasing": 0.25 * (self.diffusion_norm * torch.cos(math.pi * (1 - t)) + 1) ** 2,
            "increasing-decreasing": self.diffusion_norm * torch.sin(math.pi * (1 - t)) ** 2,
        }
        return choices[self.diffusion_coefficient_type]

    def get_time_steps(self, skip_type: str, t_T: float, t_0: float, N: int, device: torch.device) -> torch.Tensor:
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )
