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
from functools import partial

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from ..utils.dist import get_dist_size


def fsdp_wrap(
    module,
    sharding_strategy="full",
    mixed_precision=True,
    wrap_strategy="size",
    min_num_params=int(5e7),
    transformer_module=None,
    cpu_offload=False,
):
    ignored_modules = []
    for name, sub_module in module.named_modules():
        if isinstance(sub_module, nn.modules.batchnorm._BatchNorm):
            ignored_modules.append(sub_module)

    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False,
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "size":
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    world_size = get_dist_size()
    device_mesh = init_device_mesh("cuda", (world_size,))  # use Dtensor

    module = FullyShardedDataParallel(
        module,
        sharding_strategy=sharding_strategy,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        ignored_modules=ignored_modules,
        device_id=torch.cuda.current_device(),
        sync_module_states=False,  # Load ckpt on rank 0 and sync to other ranks
        limit_all_gathers=True,
        use_orig_params=True,
        device_mesh=device_mesh,
    )
    return module
