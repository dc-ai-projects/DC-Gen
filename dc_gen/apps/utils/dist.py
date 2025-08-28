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
from typing import Optional, TypeVar

T = TypeVar("T")

import torch
import torch.distributed

from ...models.utils.list import list_mean, list_sum

__all__ = [
    "dist_init",
    "is_dist_initialized",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_barrier",
    "get_dist_local_rank",
    "sync_tensor",
]


def dist_init(timeout: Optional[timedelta] = None) -> None:
    if is_dist_initialized():
        return
    try:
        torch.distributed.init_process_group(
            backend="nccl", timeout=timeout, device_id=torch.device(f"cuda:{get_dist_local_rank()}")
        )
        assert torch.distributed.is_initialized()
    except Exception as e:
        print(e)
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        print("warning: dist not init")


def is_dist_initialized() -> bool:
    return torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_dist_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if is_dist_initialized():
        torch.distributed.barrier()


def get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


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


def destroy_process_group():
    if is_dist_initialized():
        torch.distributed.destroy_process_group()
