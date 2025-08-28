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

from ...models.base import BaseT2IModel, BaseT2IModelConfig


@dataclass
class BaseT2IDiffusionModelConfig(BaseT2IModelConfig):
    adaptive_channel: bool = False

    eval_scheduler: str = MISSING
    train_scheduler: str = MISSING
    num_inference_steps: int = MISSING
    train_sampling_steps: int = 1000

    guidance_type: str = "classifier-free"
    pag_applied_layers: tuple[int] = (8,)
    interval_guidance: tuple[float] = (0.0, 1.0)
    flow_shift: float = 3.0
    use_dynamic_shifting: bool = True


class BaseT2IDiffusionModel(BaseT2IModel):
    def __init__(self, cfg: BaseT2IDiffusionModelConfig):
        super().__init__(cfg)
        self.cfg: BaseT2IDiffusionModelConfig

        if cfg.eval_scheduler == "DPMS":
            pass
        elif cfg.eval_scheduler == "SCMScheduler":
            pass  # SCMScheduler is handled in the SanaSprint model
        else:
            raise NotImplementedError(f"eval_scheduler {cfg.eval_scheduler} is not supported")

        if cfg.train_scheduler == "SanaScheduler":
            from ....c2icore.diffusioncore.models.sana_utils.sana_sampler import SanaTrainScheduler

            self.train_scheduler = SanaTrainScheduler(
                str(self.cfg.train_sampling_steps),
                noise_schedule="linear_flow",
                predict_v=True,
                learn_sigma=False,
                pred_sigma=False,
                snr=False,
                flow_shift=cfg.flow_shift,
            )
        else:
            raise NotImplementedError(f"train_scheduler {cfg.train_scheduler} is not supported")

    def forward_without_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mask=None
    ) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        raise NotImplementedError
