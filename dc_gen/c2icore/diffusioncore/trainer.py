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

from dataclasses import dataclass, field

import torch

from ...apps.utils.config import get_config
from ..trainer import C2ICoreTrainer, C2ICoreTrainerConfig
from .models.base import BaseDiffusionModel, BaseDiffusionModelConfig
from .models.dit import DiT, DiTConfig
from .models.sana_cls import SanaCls, SanaClsConfig
from .models.uvit import UViT, UViTConfig


@dataclass
class DiffusionCoreTrainerConfig(C2ICoreTrainerConfig):
    dit: DiTConfig = field(default_factory=DiTConfig)
    sana_cls: SanaClsConfig = field(default_factory=SanaClsConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)


class DiffusionCoreTrainer(C2ICoreTrainer):
    def __init__(self, cfg: DiffusionCoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: DiffusionCoreTrainerConfig
        self.model: BaseDiffusionModel

    def get_possible_models(self) -> dict[str, tuple[type[BaseDiffusionModelConfig], type[BaseDiffusionModel]]]:
        possible_models = {
            "dit": (self.cfg.dit, DiT),
            "sana_cls": (self.cfg.sana_cls, SanaCls),
            "uvit": (self.cfg.uvit, UViT),
        }
        return possible_models

    def torch_compile(self):
        if self.cfg.distributed_method == "DDP":
            self.network.forward_without_cfg = torch.compile(self.network.forward_without_cfg)
        elif self.cfg.distributed_method == "FSDP":
            self.model.module.forward_without_cfg = torch.compile(self.model.module.forward_without_cfg)
        else:
            raise ValueError(f"Distributed method {self.cfg.distributed_method} not supported")


def main():
    cfg: DiffusionCoreTrainerConfig = get_config(DiffusionCoreTrainerConfig)
    trainer = DiffusionCoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
