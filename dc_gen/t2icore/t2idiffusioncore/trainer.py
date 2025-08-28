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

from ...apps.utils.config import get_config
from ..trainer import T2ICoreTrainer, T2ICoreTrainerConfig
from .models.base import BaseT2IDiffusionModel, BaseT2IDiffusionModelConfig
from .models.sana_sprint import SanaSprint, SanaSprintConfig
from .models.sana_t2i import SanaT2I, SanaT2IConfig


@dataclass
class T2IDiffusionCoreTrainerConfig(T2ICoreTrainerConfig):
    sana_t2i: SanaT2IConfig = field(
        default_factory=lambda: SanaT2IConfig(text_encoder_name="${..text_encoders.0.name}")
    )
    sana_sprint: SanaSprintConfig = field(
        default_factory=lambda: SanaSprintConfig(text_encoder_name="${..text_encoders.0.name}")
    )

    def __post_init__(self):
        pass


class T2IDiffusionCoreTrainer(T2ICoreTrainer):
    def __init__(self, cfg: T2IDiffusionCoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: T2IDiffusionCoreTrainerConfig
        self.model: BaseT2IDiffusionModel

    def get_possible_models(self) -> dict[str, tuple[type[BaseT2IDiffusionModelConfig], type[BaseT2IDiffusionModel]]]:
        possible_models = {
            "sana_t2i": (self.cfg.sana_t2i, SanaT2I),
            "sana_sprint": (self.cfg.sana_sprint, SanaSprint),
        }
        return possible_models


def main():
    cfg: T2IDiffusionCoreTrainerConfig = get_config(T2IDiffusionCoreTrainerConfig)
    trainer = T2IDiffusionCoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
