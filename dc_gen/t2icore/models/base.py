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
import torch.nn as nn
from omegaconf import MISSING


@dataclass
class BaseT2IModelConfig:
    name: str = MISSING

    in_channels: int = 32
    input_size: int = 32

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"

    count_nfe: bool = False  # count number of function evaluations


class BaseT2IModel(nn.Module):
    def __init__(self, cfg: BaseT2IModelConfig):
        super().__init__()
        self.cfg = cfg
        self.build_model()
        self.initialize_weights()
        self.check_initialization()
        if cfg.pretrained_path is not None:
            self.load_model()
        if cfg.count_nfe:
            self.nfe = 0

    def build_model(self):
        raise NotImplementedError

    def get_trainable_modules_list(self) -> nn.ModuleList:
        raise NotImplementedError

    def load_model(self):
        raise NotImplementedError

    def initialize_weights(self):
        raise NotImplementedError

    def check_initialization(self):
        for name, param in self.named_parameters():
            if not hasattr(param, "initialized"):
                raise ValueError(f"param {name} is not initialized")
            else:
                delattr(param, "initialized")

    def enable_activation_checkpointing(self, mode: str):
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
