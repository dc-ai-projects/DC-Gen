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

# This implementation follows this blog https://bfl.ai/research/representation-comparison

import math
from typing import Optional

import numpy as np
import torch


class Flux2Scheduler:
    """Euler discrete scheduler with Flux2 shift for flow-matching inference.

    Shift formula: sigma' = (r * sigma) / (1 + (r - 1) * sigma)
    where r = sqrt(latent_seq_len / base_seq_len).
    """

    def __init__(
        self,
        num_inference_steps: int = 20,
        num_train_timesteps: int = 1000,
        base_seq_len: int = 4096,
        latent_seq_len: Optional[int] = None,
    ):
        self.num_inference_steps = num_inference_steps
        self.num_train_timesteps = num_train_timesteps
        self.base_seq_len = base_seq_len
        self.latent_seq_len = latent_seq_len

        self.sigmas: Optional[torch.Tensor] = None
        self.timesteps: Optional[torch.Tensor] = None
        self.step_index: Optional[int] = None
        self.shift_ratio: Optional[float] = None

    def get_timesteps(
        self,
        device: str | torch.device = "cpu",
        image_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute shifted sigmas and timesteps for Flux2 inference.

        Args:
            device: Target device for tensors.
            image_seq_len: Sequence length of the latent. Falls back to self.latent_seq_len,
                then self.base_seq_len (giving r=1, i.e. no shift).
        """
        seq_len = self.latent_seq_len if self.latent_seq_len is not None else image_seq_len
        if seq_len is None:
            seq_len = self.base_seq_len  # r = 1, identity
        self.shift_ratio = math.sqrt(seq_len / self.base_seq_len)

        # Raw uniform sigmas
        sigmas = np.linspace(1.0, 1.0 / self.num_inference_steps, self.num_inference_steps, dtype=np.float32)

        # Flux2 shift
        sigmas = (self.shift_ratio * sigmas) / (1.0 + (self.shift_ratio - 1.0) * sigmas)
        sigmas = sigmas.astype(np.float32)

        sigmas_t = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas_t * self.num_train_timesteps

        # Append trailing zero for the final Euler step
        self.sigmas = torch.cat([sigmas_t, torch.zeros(1, device=device)])
        self.timesteps = timesteps

        return timesteps

    def _index_for_timestep(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        indices = (self.timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float | torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler step: prev = sample + (sigma_next - sigma) * model_output."""
        step_index = self._index_for_timestep(timestep)

        sample = sample.to(torch.float32)

        sigma = self.sigmas[step_index]
        sigma_next = self.sigmas[step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        prev_sample = prev_sample.to(model_output.dtype)

        return prev_sample
