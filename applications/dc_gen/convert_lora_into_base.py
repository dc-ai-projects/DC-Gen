# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import MISSING

from dc_ai.apps.utils.config import get_config
from dc_ai.apps.utils.dtype import convert_to_dtype_recursive, get_dtype_from_str


@dataclass
class ConvertLoRAIntoBaseConfig:
    load_path: str = MISSING
    save_path: str = MISSING
    lora_type: str = "LoRA"
    use_ema: bool = True
    ema_decay: float = 0.999
    rank: int = 256
    alpha: float = 256.0
    output_dtype: Optional[str] = None


def _resolve_output_dtype(output_dtype: Optional[str]) -> Optional[torch.dtype]:
    if output_dtype is None:
        return None
    if output_dtype not in ["fp32", "bf16"]:
        raise ValueError(f"Unsupported output dtype {output_dtype!r}; expected one of: fp32, bf16")
    return get_dtype_from_str(output_dtype)


def merge_lora_weights(
    lora_state_dict: Mapping[str, torch.Tensor],
    rank: int,
    alpha: float,
    lora_type: str = "LoRA",
    device: str | torch.device = "cuda",
    output_dtype: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    resolved_output_dtype = _resolve_output_dtype(output_dtype)
    lora_pairs: OrderedDict[str, tuple[str, str]] = OrderedDict()
    merged_state_dict: dict[str, torch.Tensor] = {}

    for key in lora_state_dict:
        if ".lora_A.default" in key:
            base_key = key.replace(".lora_A.default", ".base_layer")
            paired_lora_key = key.replace(".lora_A.default", ".lora_B.default")
        elif ".lora_B.default" in key:
            base_key = key.replace(".lora_B.default", ".base_layer")
            paired_lora_key = key.replace(".lora_B.default", ".lora_A.default")
        else:
            continue
        if base_key not in lora_state_dict or paired_lora_key not in lora_state_dict:
            raise ValueError(f"Incomplete LoRA weights for {key}")

    for key in lora_state_dict:
        if "base_layer" in key:
            lora_A_key = key.replace("base_layer", "lora_A.default")
            lora_B_key = key.replace("base_layer", "lora_B.default")

            if lora_A_key in lora_state_dict:
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
        if lora_A.shape[0] != rank or lora_B.shape[1] != rank:
            raise ValueError(
                f"Configured LoRA rank {rank} does not match {lora_A_key} and {lora_B_key}: "
                f"{tuple(lora_A.shape)}, {tuple(lora_B.shape)}"
            )

        base_weight_fp32 = base_weight.to(device=device, dtype=torch.float32)
        lora_A_fp32 = lora_A.to(device=device, dtype=torch.float32)
        lora_B_fp32 = lora_B.to(device=device, dtype=torch.float32)
        if len(base_weight.shape) == 1:  # Deal with Bias
            merged_weight = base_weight_fp32 + alpha / rank * (lora_A_fp32 @ lora_B_fp32).squeeze()
        else:  # Deal with Weight
            delta_W = lora_B_fp32 @ lora_A_fp32

            if lora_type == "LoRA":
                merged_weight = base_weight_fp32 + alpha / rank * delta_W
            elif lora_type == "DoRA":
                merged_weight = base_weight_fp32 + alpha / rank * delta_W
                merged_weight = F.normalize(merged_weight, p=2, dim=1)
                lora_magnitude_key = lora_A_key.replace("lora_A", "lora_magnitude_vector")
                lora_magnitude = lora_state_dict[lora_magnitude_key].to(
                    device=device,
                    dtype=torch.float32,
                )
                merged_weight = lora_magnitude.unsqueeze(1) * merged_weight
            elif lora_type == "Mixture":
                dora_A_key = lora_A_key.replace("default", "dora_adapter")
                dora_B_key = lora_B_key.replace("default", "dora_adapter")
                dora_A = lora_state_dict[dora_A_key].to(device=device, dtype=torch.float32)
                dora_B = lora_state_dict[dora_B_key].to(device=device, dtype=torch.float32)
                merged_weight = base_weight_fp32 + alpha / rank * (dora_B @ dora_A).squeeze()
                merged_weight = F.normalize(merged_weight, p=2, dim=1)

                dora_magnitude_key = dora_A_key.replace("lora_A", "lora_magnitude_vector")
                dora_magnitude = lora_state_dict[dora_magnitude_key].to(
                    device=device,
                    dtype=torch.float32,
                )
                merged_weight = dora_magnitude.unsqueeze(1) * merged_weight
                merged_weight += alpha / rank * delta_W
            else:
                raise NotImplementedError(f"LoRA type {lora_type} is not supported")

        merged_dtype = base_weight.dtype if resolved_output_dtype is None else resolved_output_dtype
        merged_state_dict[base_key.replace(".base_layer", "")] = merged_weight.to(
            device="cpu",
            dtype=merged_dtype,
        )

    for null_embedding_key in (
        "t5_null_embedding",
        "clip_null_embedding",
        "null_text_embedding",
    ):
        merged_state_dict.pop(null_embedding_key, None)

    if resolved_output_dtype is not None:
        merged_state_dict = convert_to_dtype_recursive(
            merged_state_dict,
            resolved_output_dtype,
        )

    return merged_state_dict


def main() -> None:
    cfg = get_config(ConvertLoRAIntoBaseConfig)

    checkpoint = torch.load(
        cfg.load_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"LoRA checkpoint must contain a state dict: {cfg.load_path}")

    if cfg.use_ema:
        print("Load from EMA")
        ema_model_state_dict = checkpoint.get("ema_model_state_dict")
        if not isinstance(ema_model_state_dict, Mapping) or cfg.ema_decay not in ema_model_state_dict:
            raise ValueError(f"LoRA checkpoint must contain EMA decay {cfg.ema_decay}: {cfg.load_path}")
        lora_state_dict = ema_model_state_dict[cfg.ema_decay]
    else:
        print("Load from State Dict")
        lora_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(lora_state_dict, Mapping):
        raise ValueError(f"LoRA checkpoint does not contain the requested state dict: {cfg.load_path}")

    nonlora_checkpoint = merge_lora_weights(
        lora_state_dict,
        rank=cfg.rank,
        alpha=cfg.alpha,
        lora_type=cfg.lora_type,
        output_dtype=cfg.output_dtype,
    )
    output_dir = os.path.dirname(cfg.save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save({"model_state_dict": nonlora_checkpoint}, cfg.save_path)


if __name__ == "__main__":
    main()
