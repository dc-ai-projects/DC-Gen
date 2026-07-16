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

import time
from typing import Callable

import torch
from tqdm import tqdm


@torch.no_grad()
def test_pytorch_efficiency(func: Callable, warmup_iterations: int = 20, iterations: int = 100) -> dict[str, float]:
    for i in tqdm(range(warmup_iterations), desc="warm up"):
        func()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    for i in tqdm(range(iterations), desc="test"):
        func()
    torch.cuda.synchronize()
    end_time = time.time()

    return {
        "step_time": (end_time - start_time) / iterations,
        "throughput": iterations / (end_time - start_time),
        "memory": torch.cuda.max_memory_allocated() / 1024**3,
    }
