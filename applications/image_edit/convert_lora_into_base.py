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
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omegaconf import MISSING

from dc_ai.apps.utils.config import get_config


@dataclass
class ConvertLoRAIntoBaseConfig:
    load_path: str = MISSING
    save_path: str = MISSING
    lora_type: str = "LoRA"
    use_ema: bool = True


def merge_lora_weights(lora_state_dict, rank=128, alpha=256.0, lora_type="LoRA"):
    lora_pairs = OrderedDict()
    merged_state_dict = {}

    for key in list(lora_state_dict.keys()):
        if "base_layer" in key:
            lora_A_key = key.replace("base_layer", "lora_A.default")
            lora_B_key = key.replace("base_layer", "lora_B.default")

            if lora_A_key in lora_state_dict:
                assert lora_B_key in lora_state_dict, "Incomplete triple (BaseLayer, LoRA_A, LoRA_B)"
                lora_pairs[key] = (lora_A_key, lora_B_key)
            else:
                merged_state_dict[key.replace(".base_layer", "")] = lora_state_dict[key]
        elif "lora_A" in key or "lora_B" in key or "lora_magnitude_vector" in key:
            continue
        else:
            merged_state_dict[key] = lora_state_dict[key]
    print("Finish Loading")

    for base_key, (lora_A_key, lora_B_key) in lora_pairs.items():
        base_weight = lora_state_dict[base_key]
        lora_A = lora_state_dict[lora_A_key]
        lora_B = lora_state_dict[lora_B_key]

        if len(base_weight.shape) == 1:  # Deal with Bias
            merged_weight = base_weight + alpha / rank * (lora_A @ lora_B).squeeze()
        else:  # Deal with Weight
            delta_W = (lora_B.cuda() @ lora_A.cuda()).cpu()

            if lora_type == "LoRA":
                merged_weight = base_weight + alpha / rank * delta_W
            elif lora_type == "DoRA":
                merged_weight = base_weight + alpha / rank * delta_W
                merged_weight = F.normalize(merged_weight, p=2, dim=1)
                lora_magnitude_key = lora_A_key.replace("lora_A", "lora_magnitude_vector")
                lora_magnitude = lora_state_dict[lora_magnitude_key]
                merged_weight = lora_magnitude.unsqueeze(1) * merged_weight
            elif lora_type == "Mixture":
                dora_A_key = lora_A_key.replace("default", "dora_adapter")
                dora_B_key = lora_B_key.replace("default", "dora_adapter")
                dora_A = lora_state_dict[dora_A_key]
                dora_B = lora_state_dict[dora_B_key]
                merged_weight = base_weight + alpha / rank * (dora_B @ dora_A).squeeze()
                merged_weight = F.normalize(merged_weight, p=2, dim=1)

                dora_magnitude_key = dora_A_key.replace("lora_A", "lora_magnitude_vector")
                dora_magnitude = lora_state_dict[dora_magnitude_key]
                merged_weight = dora_magnitude.unsqueeze(1) * merged_weight
                merged_weight += alpha / rank * delta_W
            else:
                raise NotImplementedError(f"LoRA type {lora_type} is not supported.")

        merged_state_dict[base_key.replace(".base_layer", "")] = merged_weight

    return merged_state_dict


def main():
    cfg = get_config(ConvertLoRAIntoBaseConfig)
    load_path = cfg.load_path
    save_path = cfg.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = torch.load(load_path)
    if cfg.use_ema:
        print("Load from EMA")
        checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
    else:
        print("Load from State Dict")
        checkpoint = checkpoint["model_state_dict"]

    nonlora_checkpoint = merge_lora_weights(checkpoint, rank=256, alpha=256.0, lora_type=cfg.lora_type)
    torch.save(nonlora_checkpoint, save_path)


if __name__ == "__main__":
    main()


"""
python -m applications.dc_videogen.convert_lora_into_base
"""
