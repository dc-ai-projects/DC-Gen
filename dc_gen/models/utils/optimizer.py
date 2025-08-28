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

from torch.optim import Optimizer


def get_optimizer_params_num(optimizer: Optimizer, unit: float = 1e6) -> float:
    num_params = 0
    state_dict = optimizer.state_dict()["state"]
    for idx in state_dict:
        for _, value in state_dict[idx].items():
            num_params += value.numel()
    return num_params / unit
