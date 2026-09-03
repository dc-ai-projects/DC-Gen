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

from omegaconf import MISSING
from torch.utils.data import Dataset

from ...apps.utils.video import MultiResolutionVideoDataset, write_video
from .base import STAECoreDataProvider, STAECoreDataProviderConfig


@dataclass
class ActivityNetDataProviderConfig(STAECoreDataProviderConfig):
    name: str = "ActivityNet"
    data_dir: str = MISSING


@dataclass
class ActivityNet12TestDataProviderConfig(ActivityNetDataProviderConfig):
    name: str = "ActivityNet12Test"
    size_transform: str = "TemporalCenterCropSpatialDMCrop"
    metadata_path: Optional[str] = "assets/data/examination/ActivityNet12Test.csv"
    data_dir: str = "~/dataset/activitynet/v1-2/test"


@dataclass
class ActivityNet13TestDataProviderConfig(ActivityNetDataProviderConfig):
    name: str = "ActivityNet13Test"
    size_transform: str = "TemporalCenterCropSpatialDMCrop"
    metadata_path: Optional[str] = "assets/data/examination/ActivityNet13Test.csv"
    data_dir: str = "~/dataset/activitynet/v1-3/test"

    def __post_init__(self):
        self.fvd_ref_path: str = f"assets/data/fvd/activitynet_1_3_test_{self.h}_{self.w}_{self.t}_{self.fps}.npz"


@dataclass
class ActivityNet13TestLongVideoDataProviderConfig(ActivityNetDataProviderConfig):
    name: str = "ActivityNet13TestLongVideo"
    size_transform: str = "TemporalCenterCropSpatialDMCrop"
    metadata_path: Optional[str] = "assets/data/examination/ActivityNet13Test.csv"
    data_dir: str = "~/dataset/activitynet/v1-3/test"
    min_t: Optional[int] = 1000
    end_index: Optional[int] = 1000


class ActivityNetDataProvider(STAECoreDataProvider):
    def __init__(self, cfg: ActivityNetDataProviderConfig):
        super().__init__(cfg)
        self.cfg: ActivityNetDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = MultiResolutionVideoDataset(self.cfg.data_dir, size_transform, transform, self.metadata)
        return dataset
