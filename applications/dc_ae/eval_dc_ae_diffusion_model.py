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

from omegaconf import MISSING

from dc_gen.apps.utils.config import get_config
from dc_gen.c2icore.diffusioncore.trainer import DiffusionCoreTrainer, DiffusionCoreTrainerConfig


@dataclass
class EvalDiffusionModelConfig(DiffusionCoreTrainerConfig):
    dataset: str = MISSING
    model: str = MISSING


def main():
    cfg = get_config(EvalDiffusionModelConfig)

    if cfg.dataset == "imagenet_512":
        cfg.resolution = 512
        cfg.eval_data_provider = "sample_class"
        cfg.sample_class.batch_size = 32
        cfg.fid.ref_path = "assets/data/fid/imagenet_train_512.npz"
    else:
        raise NotImplementedError

    trainer = DiffusionCoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
