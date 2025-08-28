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

from omegaconf import MISSING, OmegaConf

from dc_gen.aecore.trainer import AECoreTrainer, AECoreTrainerConfig
from dc_gen.apps.utils.config import get_config


@dataclass
class EvalAEModelConfig:
    dataset: str = MISSING
    model: str = MISSING
    amp: str = "fp32"
    pretrained_path: Optional[str] = None
    run_dir: str = MISSING


def main():
    cfg = get_config(EvalAEModelConfig)

    trainer_cfg: AECoreTrainerConfig = OmegaConf.structured(AECoreTrainerConfig)
    trainer_cfg.run_dir = cfg.run_dir
    if cfg.dataset == "ImageNet_512":
        trainer_cfg.eval_data_providers = ("ImageNet_512",)
        trainer_cfg.base_sample_size = 512
        trainer_cfg.base_batch_size = 64
    else:
        raise NotImplementedError
    trainer_cfg.model = cfg.model
    trainer_cfg.amp = cfg.amp
    trainer_cfg = OmegaConf.to_object(trainer_cfg)

    trainer = AECoreTrainer(trainer_cfg)
    trainer.run()


if __name__ == "__main__":
    main()
