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

import torch
from torch.utils.data import Dataset

from .base import C2ICoreDataProvider, C2ICoreDataProviderConfig

__all__ = ["SampleClassDataProviderConfig", "SampleClassDataset", "SampleClassDataProvider"]


@dataclass
class SampleClassDataProviderConfig(C2ICoreDataProviderConfig):
    name: str = "SampleClass"
    num_classes: int = 1000
    num_samples: int = 50000
    seed: int = 0


class SampleClassDataset(Dataset):
    def __init__(self, cfg: SampleClassDataProviderConfig):
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        if cfg.num_classes > 0:
            self.class_ids = torch.randint(0, cfg.num_classes, (cfg.num_samples,), generator=self.generator).int()
        else:
            self.class_ids = -torch.ones(cfg.num_samples).int()

    def __len__(self):
        return self.cfg.num_samples

    def __getitem__(self, index):
        return self.class_ids[index], self.cfg.num_classes


class SampleClassDataProvider(C2ICoreDataProvider):
    def __init__(self, cfg: SampleClassDataProviderConfig):
        super().__init__(cfg)
        self.cfg: SampleClassDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        return SampleClassDataset(self.cfg)
