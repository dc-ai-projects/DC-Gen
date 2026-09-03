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
from datetime import timedelta
from typing import Optional, TypeVar, cast

T = TypeVar("T")
ScalarT = TypeVar("ScalarT", int, float)

import torch
import torch.distributed

from ...models.utils.list import list_mean, list_sum

_DISTRIBUTED_ENV_VARS = ("RANK", "WORLD_SIZE", "LOCAL_RANK")

__all__ = [
    "dist_init",
    "is_dist_initialized",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "all_reduce_sum",
    "sync_tensor",
]


def dist_init(timeout: Optional[timedelta] = None) -> None:
    # Shared entrypoints may call this more than once, so initialization must be idempotent.
    if is_dist_initialized():
        return

    # No launcher-provided rank metadata means this is a normal standalone process.
    present_env_vars = [name for name in _DISTRIBUTED_ENV_VARS if name in os.environ]
    if not present_env_vars:
        return

    # Partial metadata cannot identify this process reliably; fail before initializing NCCL.
    missing_env_vars = [name for name in _DISTRIBUTED_ENV_VARS if name not in os.environ]
    if missing_env_vars:
        raise RuntimeError(f"Incomplete distributed environment: missing {', '.join(missing_env_vars)}.")

    # A complete launcher environment assigns each process a rank and one local CUDA device.
    rank = get_dist_rank()
    local_rank = get_dist_local_rank()
    dist_size = get_dist_size()
    try:
        torch.distributed.init_process_group(
            backend="nccl", timeout=timeout, device_id=torch.device(f"cuda:{local_rank}")
        )
        if not is_dist_initialized():
            raise RuntimeError("init_process_group() returned without initializing torch.distributed")
    except Exception as error:
        # Never disguise a failed distributed process as standalone; preserve its rank identity.
        raise RuntimeError(
            "Failed to initialize distributed process group "
            f"(rank={rank}, local_rank={local_rank}, world_size={dist_size})."
        ) from error


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def all_reduce_sum(value: ScalarT) -> ScalarT:
    if type(value) is int:
        value_type = int
    elif type(value) is float:
        value_type = float
    else:
        raise TypeError(f"all_reduce_sum expected int or float, got {type(value).__name__}")

    if not is_dist_initialized():
        return value

    value_tensor = torch.tensor([value], device=torch.device("cuda"))
    torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.SUM)
    return cast(ScalarT, value_type(value_tensor.item()))


# Warning: This function is not differentiable. Use torch.distributed.nn.all_gather if you need gradients.
def sync_tensor(tensor: torch.Tensor | float, reduce="mean") -> torch.Tensor | list[torch.Tensor]:
    if not is_dist_initialized():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.Tensor(1).fill_(tensor).cuda()
    tensor_list = [torch.empty_like(tensor) for _ in range(get_dist_size())]
    torch.distributed.all_gather(tensor_list, tensor.contiguous(), async_op=False)
    if reduce == "mean":
        return list_mean(tensor_list)
    elif reduce == "sum":
        return list_sum(tensor_list)
    elif reduce == "cat":
        return torch.cat(tensor_list, dim=0)
    elif reduce == "root":
        return tensor_list[0]
    else:
        return tensor_list


def sync_object(obj: T) -> list[T]:
    if not is_dist_initialized():
        return [obj]
    obj_list = [None for _ in range(get_dist_size())]
    torch.distributed.all_gather_object(obj_list, obj)
    return obj_list


def broadcast_object(obj: T, src: int = 0) -> T:
    if not is_dist_initialized():
        return obj
    obj_list = [obj]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def gather_object(obj: T, dst: int = 0) -> Optional[list[T]]:
    if not is_dist_initialized():
        return [obj]
    obj_list = [None for _ in range(get_dist_size())] if get_dist_rank() == dst else None
    torch.distributed.gather_object(obj, obj_list, dst=dst)
    return obj_list


def gather_dict(d: dict, dst: int = 0) -> Optional[dict]:
    if not is_dist_initialized():
        return d
    dict_list = [None for _ in range(get_dist_size())] if get_dist_rank() == dst else None
    torch.distributed.gather_object(d, dict_list, dst=dst)
    if get_dist_rank() == dst:
        all_keys = [key for d in dict_list for key in d]
        assert len(all_keys) == len(set(all_keys)), f"require unique keys, but found {all_keys}"
        return {key: value for d in dict_list for key, value in d.items()}
    else:
        return None


def gather_list(l: list, dst: int = 0) -> Optional[list]:
    if not is_dist_initialized():
        return l
    l_list = [None for _ in range(get_dist_size())] if get_dist_rank() == dst else None
    torch.distributed.gather_object(l, l_list, dst=dst)
    if get_dist_rank() == dst:
        return [item for l in l_list for item in l]
    else:
        return None


def destroy_process_group():
    if is_dist_initialized():
        torch.distributed.destroy_process_group()


def distribute_list_to_rank(data_list: list[T]) -> list[T]:
    data_list = data_list[get_dist_rank() :: get_dist_size()]
    return data_list
