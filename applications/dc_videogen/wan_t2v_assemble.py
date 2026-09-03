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
from typing import Any, Optional

import torch

from dc_ai.apps.utils.config import get_config
from dc_ai.models.utils.network import get_submodule_weights
from dc_ai.videogencore.models.wan_t2v import WanT2V, WanT2VConfig


def load_checkpoint(path: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "ema_model_state_dict" in checkpoint:
        checkpoint = checkpoint["ema_model_state_dict"]
        first_value = next(iter(checkpoint.values()))
        if isinstance(first_value, dict):
            checkpoint = first_value
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # remove prefix 0.
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith("0."):
            key = key[2:]
        state_dict[key] = value

    return state_dict


@dataclass
class WanT2VAssembleConfig(WanT2VConfig):
    backbone_checkpoint_path: Optional[str] = None
    patch_embedding_checkpoint_path: Optional[str] = None
    head_checkpoint_path: Optional[str] = None

    checkpoint_export_path: str = "test.pt"


def assemble_checkpoint(cfg: WanT2VAssembleConfig) -> WanT2V:
    target_model = WanT2V(cfg)

    # load backbone weights
    checkpoint = load_checkpoint(cfg.backbone_checkpoint_path)
    for key, target in zip(
        ["time_embedding", "time_projection", "text_embedding", "blocks", "head.norm"],
        [
            target_model.time_embedding,
            target_model.time_projection,
            target_model.text_embedding,
            target_model.blocks,
            target_model.head.norm,
        ],
    ):
        target.load_state_dict(get_submodule_weights(checkpoint, f"{key}."))
        print(f"Loaded {key} from {cfg.backbone_checkpoint_path}")

    target_model.head.modulation = torch.nn.Parameter(checkpoint["head.modulation"])

    # load patch embedding if given patch_embedding_checkpoint_path
    if cfg.patch_embedding_checkpoint_path is not None:
        checkpoint = load_checkpoint(cfg.patch_embedding_checkpoint_path)
        patch_embedding_weights = get_submodule_weights(checkpoint, "patch_embedding.")
        target_model.patch_embedding.load_state_dict(patch_embedding_weights)
        print(f"Loaded patch_embedding from {cfg.patch_embedding_checkpoint_path}")

    return target_model


def main():
    cfg = get_config(WanT2VAssembleConfig)
    target_model = assemble_checkpoint(cfg)
    os.makedirs(os.path.dirname(cfg.checkpoint_export_path), exist_ok=True)
    torch.save(target_model.state_dict(), cfg.checkpoint_export_path)
    print(f"Assembled checkpoint saved to {cfg.checkpoint_export_path}")
    print(target_model)


if __name__ == "__main__":
    main()
