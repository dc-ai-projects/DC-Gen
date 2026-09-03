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
from safetensors.torch import load_file


@dataclass
class ConvertCheckpointConfig:
    model: str = "wan_t2v"
    file_dir: str = "assets/checkpoints/t2v/1.3b"

    load_filenames: tuple[str, ...] = ("diffusion_pytorch_model.safetensors",)
    save_filename: str = MISSING


def convert_checkpoint(cfg: ConvertCheckpointConfig):
    load_paths = [os.path.join(cfg.file_dir, filename) for filename in cfg.load_filenames]
    save_path = os.path.join(cfg.file_dir, cfg.save_filename)

    checkpoint = {}

    for load_path in load_paths:
        if load_path.endswith(".safetensors"):
            cur_checkpoint = load_file(load_path)
        else:
            raise NotImplementedError("Unrecognized checkpoint type")

        for key, value in cur_checkpoint.items():
            assert key not in checkpoint, f"Duplicate keys {key}"
            checkpoint[key] = value

    if cfg.model in ["wan_t2v_1.3B", "wan_t2v_14B", "wan_t2v_A14B", "wan_ti2v_5B"]:
        null_embedding = torch.load("assets/data/null_text_embeddings/wan2.1-t2v/umt5-512-bf16.pth")
        checkpoint["null_embedding"] = null_embedding
    elif cfg.model in ["wan_i2v_14B", "wan_i2v_A14B"]:
        null_embedding = torch.load("assets/data/null_text_embeddings/wan2.1-i2v/umt5-512-bf16.pth")
        checkpoint["null_embedding"] = null_embedding
    else:
        raise NotImplementedError(f"{cfg.model} is not supported.")

    assert save_path.endswith(".pt") or save_path.endswith(".pth"), "Must save model as checkpoint"
    torch.save(checkpoint, save_path)


def main():
    cfg: ConvertCheckpointConfig = OmegaConf.merge(OmegaConf.structured(ConvertCheckpointConfig), OmegaConf.from_cli())
    convert_checkpoint(cfg)


if __name__ == "__main__":
    main()

"""
python -m applications.dc_videogen.convert_checkpoint model=wan_t2v_1.3B \
    save_filename=wan_t2v_1.3b_pretrained.pt
"""
