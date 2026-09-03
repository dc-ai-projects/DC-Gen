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
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import MISSING

from dc_ai.apps.utils.config import get_config
from dc_ai.models.utils.network import get_submodule_weights
from dc_ai.t2icore.models.zimage import ZImage, ZImageConfig


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Z-Image checkpoint must contain a state dict: {path}")

    if "ema_model_state_dict" in checkpoint:
        checkpoint = checkpoint["ema_model_state_dict"]
        if not isinstance(checkpoint, Mapping) or not checkpoint:
            raise ValueError(f"Z-Image EMA checkpoint must contain a state dict: {path}")
        first_value = next(iter(checkpoint.values()))
        if isinstance(first_value, Mapping):
            checkpoint = first_value
    elif "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Z-Image checkpoint must contain a state dict: {path}")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Z-Image checkpoint contains a non-tensor value for key {key!r}: {path}")
        if key.startswith("0."):
            key = key[2:]
        state_dict[key] = value

    transformer_state_dict = get_submodule_weights(state_dict, "transformer.")
    if transformer_state_dict:
        return transformer_state_dict
    return state_dict


@dataclass
class ZImageAssembleConfig(ZImageConfig):
    backbone_checkpoint_path: str = MISSING
    patch_embedding_checkpoint_path: Optional[str] = None
    head_checkpoint_path: Optional[str] = None

    checkpoint_export_path: str = MISSING


def assemble_checkpoint(cfg: ZImageAssembleConfig) -> ZImage:
    target_model = ZImage(cfg)
    target_transformer = target_model.transformer

    checkpoint = load_checkpoint(cfg.backbone_checkpoint_path)

    excluded_modules = {"all_x_embedder", "all_final_layer"}
    transformer_modules = dict(target_transformer.named_children())
    supported_roots = set(transformer_modules) | {"x_pad_token", "cap_pad_token"}
    unsupported_roots = {key.partition(".")[0] for key in checkpoint if key.partition(".")[0] not in supported_roots}
    if unsupported_roots:
        raise ValueError(f"Z-Image backbone checkpoint contains unsupported modules: {sorted(unsupported_roots)}")

    for name, module in transformer_modules.items():
        if name in excluded_modules:
            continue
        module.load_state_dict(
            get_submodule_weights(checkpoint, f"{name}."),
            strict=True,
        )
        print(f"Loaded {name} from {cfg.backbone_checkpoint_path}")

    with torch.no_grad():
        for name in ("x_pad_token", "cap_pad_token"):
            if name not in checkpoint:
                raise ValueError(f"Z-Image backbone checkpoint is missing {name}: {cfg.backbone_checkpoint_path}")
            target = getattr(target_transformer, name)
            source = checkpoint[name]
            if target.shape != source.shape:
                raise ValueError(
                    f"Z-Image backbone {name} shape mismatch: expected {tuple(target.shape)}, got {tuple(source.shape)}"
                )
            target.copy_(source)

    # load patch embedding if given patch_embedding_checkpoint_path
    if cfg.patch_embedding_checkpoint_path is not None:
        checkpoint = load_checkpoint(cfg.patch_embedding_checkpoint_path)

        x_embedder_weights = get_submodule_weights(checkpoint, "x_embedder.")
        embedder_key = f"{cfg.patch_size}-1"
        target_transformer.all_x_embedder[embedder_key].load_state_dict(x_embedder_weights, strict=True)
        print(f"Loaded x_embedder from {cfg.patch_embedding_checkpoint_path}")

    # load head weights if given head_checkpoint_path
    if cfg.head_checkpoint_path is not None:
        checkpoint = load_checkpoint(cfg.head_checkpoint_path)
        head_weights = get_submodule_weights(checkpoint, "final_layer.")
        final_layer_key = f"{cfg.patch_size}-1"
        target_transformer.all_final_layer[final_layer_key].load_state_dict(head_weights, strict=True)
        print(f"Loaded head from {cfg.head_checkpoint_path}")

    return target_model


def build_dc_gen_checkpoint(model: ZImage) -> dict[str, dict[str, torch.Tensor]]:
    """Wrap an assembled Z-Image model in the DC-Gen model-only checkpoint layout."""

    return {"model_state_dict": dict(model.state_dict())}


def main():
    cfg = get_config(ZImageAssembleConfig)

    target_model = assemble_checkpoint(cfg)
    output_dir = os.path.dirname(cfg.checkpoint_export_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(build_dc_gen_checkpoint(target_model), cfg.checkpoint_export_path)
    print(f"Assembled checkpoint saved to {cfg.checkpoint_export_path}")
    print(target_model)


if __name__ == "__main__":
    main()
