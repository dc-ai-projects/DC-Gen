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

import collections
import os
from inspect import signature
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "is_parallel",
    "get_device",
    "get_params_num",
    "get_same_padding",
    "resize",
    "build_kwargs_from_config",
    "load_state_dict_from_file",
    "get_submodule_weights",
    "freeze_weights",
    "to_inference_mode",
]


def is_parallel(model: nn.Module) -> bool:
    return isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def get_device(model: nn.Module) -> torch.device:
    return model.parameters().__next__().device


def get_dtype(model: nn.Module) -> torch.dtype:
    return model.parameters().__next__().dtype


def get_params_num(model: nn.Module, unit: float = 1e6, train_only=True) -> float:
    n_params = 0
    for p in model.parameters():
        if train_only and not p.requires_grad:
            continue
        n_params += p.numel()
    return n_params / unit


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resize(
    x: torch.Tensor,
    size: Optional[Any] = None,
    scale_factor: Optional[list[float]] = None,
    mode: str = "bicubic",
    align_corners: Optional[bool] = False,
) -> torch.Tensor:
    if mode in {"bilinear", "bicubic"}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )
    elif mode in {"nearest", "area"}:
        return F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    else:
        raise NotImplementedError(f"resize(mode={mode}) not implemented.")


def build_kwargs_from_config(config: dict, target_func: Callable) -> dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    file = os.path.realpath(os.path.expanduser(file))
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights


def get_model_weights_sum(model: nn.Module):
    return sum(parameter.sum().item() for parameter in model.parameters())


def get_dtype_from_str(dtype: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    raise NotImplementedError(f"dtype {dtype} is not supported")


def freeze_weights(model: nn.Module, freeze_bias: bool = True) -> None:
    if not freeze_bias:
        for name, parameter in model.named_parameters():
            if "bias" not in name:
                parameter.requires_grad = False
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False


def to_inference_mode(model: nn.Module) -> nn.Module:
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model
