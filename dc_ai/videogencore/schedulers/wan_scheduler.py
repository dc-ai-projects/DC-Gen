# Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# Convert unipc for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

from typing import Union

import torch
from torch import nn

from .fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanScheduler:
    def __init__(
        self,
        num_train_steps: int = 1000,
    ):
        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_steps,
            shift=1,
        )

    def set_timesteps(self, num_inference_steps: int = 50, flow_shift: float = 5.0, device: str = "cuda"):
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=flow_shift)
        timesteps = self.scheduler.timesteps
        return timesteps

    def step(
        self,
        model: nn.Module,
        latents: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        text_embeddings: torch.FloatTensor,
        null_text_embeddings: torch.FloatTensor,
        cfg_scale: float = 5.0,
        **kwargs,
    ):
        timestep = timestep.expand(latents.shape[0])

        noise_pred_cond = model(
            x=latents,
            t=timestep,
            y=text_embeddings,
            **kwargs,
        )
        noise_pred_uncond = model(
            x=latents,
            t=timestep,
            y=null_text_embeddings,
            **kwargs,
        )

        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        latents = self.scheduler.step(noise_pred, timestep, latents)

        return latents
