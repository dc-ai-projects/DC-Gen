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
from omegaconf import MISSING

from .base import BaseVideoGenModel, BaseVideoGenModelConfig


@dataclass
class BaseVideoGenDiffusionModelConfig(BaseVideoGenModelConfig):
    eval_scheduler: str = MISSING
    train_scheduler: str = MISSING
    num_inference_steps: int = MISSING
    train_sampling_steps: int = 1000

    pag_applied_layers: Optional[tuple[int]] = None

    guidance_type: str = "classifier-free"
    interval_guidance: tuple[float, float] = (0.0, 1.0)
    flow_shift: float = 3.0


class BaseVideoGenDiffusionModel(BaseVideoGenModel):
    def __init__(self, cfg: BaseVideoGenDiffusionModelConfig):
        super().__init__(cfg)
        self.cfg: BaseVideoGenDiffusionModelConfig

        if cfg.eval_scheduler == "DPMS":
            pass
        elif cfg.eval_scheduler == "WanScheduler":
            from ..schedulers.wan_scheduler import WanScheduler

            self.eval_scheduler = WanScheduler()
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

        if cfg.train_scheduler == "FlowMatchScheduler":
            from ..schedulers.flow_match_scheduler import FlowMatchScheduler

            self.training_scheduler = FlowMatchScheduler(
                num_inference_steps=self.cfg.num_inference_steps,
                shift=self.cfg.flow_shift,
            )
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        raise NotImplementedError
