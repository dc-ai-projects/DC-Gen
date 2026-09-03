# Modified from https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/schedulers/flow_match.py.
# The original implementation is by Zhongjie Duan, licensed under the Apache License 2.0.

import math

import torch


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        sigma_min = 0.0
        extra_one_step = True
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal

        self.linear_timesteps_weights = None
        self.training = False
        self.set_timesteps(num_inference_steps, training=True)

    def set_timesteps(
        self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None
    ):
        if shift is not None:
            self.shift = shift

        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength

        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)

        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            mu = self.calculate_shift(dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)

        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)

        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas

        self.sigmas = self.sigmas.cuda()
        self.timesteps = self.sigmas * self.num_train_timesteps

        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - self.num_train_timesteps / 2) / self.num_train_timesteps) ** 2)
            y_shifted = y - y.min()
            self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
            self.training = True
        else:
            self.training = False

    def training_losses(self, model, x_start, timesteps, model_kwargs=None):
        noise = torch.randn_like(x_start)

        noisy_samples = self.add_noise(x_start, noise, timesteps)

        target = noise - x_start

        if model_kwargs is None:
            model_output = model(noisy_samples, timesteps)
        else:
            model_output = model(noisy_samples, timesteps, **model_kwargs)

        loss = (model_output - target) ** 2

        timestep_id = self.get_timestep_id(timesteps)
        weights = self.linear_timesteps_weights[timestep_id].view(-1, 1, 1, 1, 1)
        weighted_loss = loss * weights

        return {
            "loss": weighted_loss.mean(),
        }

    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def get_timestep_id(self, timestep):
        timestep_id = torch.argmin((self.timesteps - timestep.unsqueeze(-1)).abs(), dim=1)
        return timestep_id

    def add_noise(self, original_samples, noise, timestep):
        timestep_id = self.get_timestep_id(timestep)
        sigma = self.sigmas[timestep_id].view(-1, 1, 1, 1, 1)
        sample = (1.0 - sigma) * original_samples + sigma * noise
        return sample
