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

import os
from dataclasses import dataclass

import torch
from omegaconf import MISSING, OmegaConf


@dataclass
class ConvertCheckpointConfig:
    model: str = "flux"
    save_path: str = MISSING


def convert_checkpoint(cfg: ConvertCheckpointConfig):

    if cfg.model == "flux":
        from diffusers import FluxPipeline

        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        checkpoint = pipeline.transformer.state_dict()

    elif cfg.model == "flux_krea":
        from diffusers import FluxPipeline

        pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-Krea-dev", torch_dtype=torch.float16)
        checkpoint = pipeline.transformer.state_dict()

    elif cfg.model == "z_image_turbo":
        from diffusers import ZImagePipeline

        pipeline = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        checkpoint = pipeline.transformer.state_dict()

    else:
        raise NotImplementedError(f"Model {cfg.model} is not supported.")

    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    torch.save(checkpoint, cfg.save_path)
    print(f"Checkpoint successfully saved at {cfg.save_path}.")


def main():
    cfg: ConvertCheckpointConfig = OmegaConf.merge(OmegaConf.structured(ConvertCheckpointConfig), OmegaConf.from_cli())
    convert_checkpoint(cfg)


if __name__ == "__main__":
    main()
