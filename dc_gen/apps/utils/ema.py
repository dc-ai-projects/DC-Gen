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

import copy
import math
from itertools import chain

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel

from ...models.utils import is_parallel

__all__ = ["EMA"]


def update_ema(ema: nn.Module, new_state_dict: dict[str, torch.Tensor], decay: float) -> None:
    for k, v in ema.state_dict().items():
        if v.dtype.is_floating_point:
            v -= (1.0 - decay) * (v - new_state_dict[k].detach())


class EMA:
    def __init__(self, model: nn.Module, decay_list: list[float], warmup_steps=2000):
        if isinstance(decay_list, float):
            decay_list = [decay_list]
        self.decay_list = decay_list
        self.shadows = {
            decay: copy.deepcopy(model.module if is_parallel(model) else model).eval() for decay in decay_list
        }
        self.warmup_steps = warmup_steps

        for p in chain(*[ema.parameters() for ema in self.shadows.values()]):
            p.requires_grad = False

    def step(self, model: nn.Module, global_step: int) -> None:
        with torch.inference_mode():
            msd = (model.module if is_parallel(model) else model).state_dict()
            for decay, ema in self.shadows.items():
                update_ema(ema, msd, decay * (1 - math.exp(-global_step / self.warmup_steps)))

    def state_dict(self) -> dict[float, dict[str, torch.Tensor]]:
        return {decay: ema.state_dict() for decay, ema in self.shadows.items()}

    def load_state_dict(self, state_dict: dict[float, dict[str, torch.Tensor]]) -> None:
        for decay in state_dict:
            if decay in self.shadows:
                self.shadows[decay].load_state_dict(state_dict[decay])


def update_ema_fsdp(ema: FullyShardedDataParallel, model: FullyShardedDataParallel, decay: float) -> None:
    for ema_param, model_param in zip(ema.parameters(), model.parameters()):
        if ema_param.dtype.is_floating_point:
            ema_param -= (1.0 - decay) * (ema_param - model_param.detach())


class EMA_fsdp:
    def __init__(self, shadows: dict[float, FullyShardedDataParallel], warmup_steps=2000):
        self.decay_list = list(shadows.keys())
        self.shadows = shadows
        self.warmup_steps = warmup_steps

    def step(self, model: FullyShardedDataParallel, global_step: int) -> None:
        with torch.inference_mode():
            for decay, ema in self.shadows.items():
                update_ema_fsdp(ema, model, decay * (1 - math.exp(-global_step / self.warmup_steps)))

    def state_dict(self) -> dict[float, dict[str, torch.Tensor]]:
        return {decay: get_model_state_dict(ema) for decay, ema in self.shadows.items()}

    def load_state_dict(self, state_dict: dict[float, dict[str, torch.Tensor]]) -> None:
        for decay in state_dict:
            set_model_state_dict(self.shadows[decay], state_dict[decay], options=StateDictOptions(strict=True))
