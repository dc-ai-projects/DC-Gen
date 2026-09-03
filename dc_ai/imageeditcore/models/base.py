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
class BaseImageEditModelConfig:
    name: str = MISSING

    in_channels: int = 32
    input_size: int = 32

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"
    load_before_lora: bool = True

    count_nfe: bool = False  # count number of function evaluations

    # lora
    use_lora: bool = False
    lora_rank: int = 256
    lora_alpha: int = 256


class BaseImageEditModel(nn.Module):
    def __init__(self, cfg: BaseImageEditModelConfig):
        super().__init__()
        self.cfg = cfg
        self.build_model()
        self.initialize_weights()
        self.check_initialization()

        if cfg.load_before_lora:
            if cfg.pretrained_path is not None:
                self.load_model()

        if self.cfg.use_lora:
            self.build_lora()

        if not cfg.load_before_lora:
            if cfg.pretrained_path is not None:
                self.load_model()

        if cfg.count_nfe:
            self.nfe = 0

    def build_model(self):
        raise NotImplementedError

    def get_lora_target_modules(self) -> set[str]:
        raise NotImplementedError

    def build_lora(self):
        from peft import LoraConfig, inject_adapter_in_model

        target_modules = self.get_lora_target_modules()
        lora_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
            bias="none",
        )

        self = inject_adapter_in_model(lora_config, self)

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
        neg_text_embed_info: dict[str, dict[str, torch.Tensor]],
        image_embed_info: dict[str, torch.Tensor],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, func_name: str, *args, **kwargs):
        return getattr(self, func_name)(*args, **kwargs)
