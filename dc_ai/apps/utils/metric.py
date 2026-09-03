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

from .dist import all_reduce_sum

__all__ = ["AverageMeter"]


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, delta_n: int = 1) -> float:
        delta_count = all_reduce_sum(delta_n)
        delta_sum = all_reduce_sum(val * delta_n)
        self.count += delta_count
        self.sum += delta_sum

        return -1.0 if delta_count == 0 else delta_sum / delta_count

    def get_count(self) -> int:
        return self.count

    @property
    def avg(self) -> float:
        return -1.0 if self.count == 0 else self.sum / self.count
