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
from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from ...apps.utils.video import MultiResolutionVideoFolder, write_video
from .base import STAECoreDataProvider, STAECoreDataProviderConfig


@dataclass
class UCF101DataProviderConfig(STAECoreDataProviderConfig):
    name: str = "UCF101"
    data_dir: str = "~/dataset/ucf101/mp4"
    metadata_path: str = "assets/data/examination/UCF101.csv"
    split: Optional[str] = None


class UCF101DataProvider(STAECoreDataProvider):
    def __init__(self, cfg: UCF101DataProviderConfig):
        super().__init__(cfg)
        self.cfg: UCF101DataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = MultiResolutionVideoFolder(
            self.cfg.data_dir, size_transform, transform, return_dict=True, metadata=self.metadata
        )
        return dataset

    def build_dataset_mask(self, complete_dataset: Dataset) -> np.ndarray:
        mask = super().build_dataset_mask(complete_dataset)
        if self.cfg.split == "eval":
            mask = mask & np.array(self.metadata["eval"])
        elif self.cfg.split == "train":
            mask = mask & (~np.array(self.metadata["eval"]))
        else:
            raise ValueError(f"split {self.cfg.split} is not supported")
        return mask
