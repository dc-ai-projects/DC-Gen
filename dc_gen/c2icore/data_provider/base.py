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

from dataclasses import dataclass

import numpy as np
from torch.utils.data import Dataset, Sampler

from ...apps.data_provider.dc_base import BaseDataProvider, BaseDataProviderConfig
from ...apps.data_provider.sampler import DistributedRangedSampler

__all__ = ["C2ICoreDataProviderConfig", "C2ICoreDataProvider"]


@dataclass
class C2ICoreDataProviderConfig(BaseDataProviderConfig):
    pass


class C2ICoreDataProvider(BaseDataProvider):
    def __init__(self, cfg: C2ICoreDataProviderConfig):
        super().__init__(cfg)
        self.cfg: C2ICoreDataProviderConfig

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise NotImplementedError

    def build_sampler(self) -> Sampler:
        return DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
        )
