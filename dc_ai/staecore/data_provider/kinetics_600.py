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
from typing import Any, Callable, Optional

import numpy as np
import pandas
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset

from ...apps.data_provider.web_dataset.wids import WebDataset
from ...apps.utils.image import IdentityTransform
from ...apps.utils.video import VideoLoader, parse_index, write_video
from .base import STAECoreDataProvider, STAECoreDataProviderConfig


@dataclass
class Kinetics600DataProviderConfig(STAECoreDataProviderConfig):
    name: str = "Kinetics600"
    data_dir: str = MISSING
    wds_meta_path: Optional[str] = MISSING
    metadata_path: Optional[str] = MISSING


@dataclass
class Kinetics600TestDataProviderConfig(Kinetics600DataProviderConfig):
    name: str = "Kinetics600Test"
    data_dir: str = "~/dataset/kinetics_600/test"
    wds_meta_path: Optional[str] = "assets/data/meta/kinetics_600_test.json"
    metadata_path: Optional[str] = "assets/data/examination/Kinetics600Test.csv"
    size_transform: str = "TemporalCenterCropSpatialDMCrop"


@dataclass
class Kinetics600Test5000DataProviderConfig(Kinetics600TestDataProviderConfig):
    end_index: Optional[int] = 5001

    def __post_init__(self):
        self.fvd_ref_path: str = f"assets/data/fvd/kinetics_600_test_5000_{self.h}_{self.w}_{self.t}_{self.fps}.npz"


class Kinetics600Dataset(WebDataset):
    def __init__(
        self,
        data_dir: str,
        meta_path: str,
        size_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        metadata: Optional[pandas.DataFrame] = None,
    ):
        super().__init__(data_dir, meta_path)
        self.size_transform = size_transform
        self.transform = transform
        self.metadata = metadata

    def __getitem__(self, index: int | tuple[int, int, Optional[float], int, int]) -> dict[str, Any]:
        index, h, w, t, fps, seed = parse_index(index)

        sample = self.dataset[index]
        video = VideoLoader(sample.pop(".mp4"))
        sample.pop("__dataset__", None)
        if self.size_transform is not None and not isinstance(self.size_transform, IdentityTransform):
            video = self.size_transform(
                video, self.metadata.at[index, "T"], self.metadata.at[index, "fps"], h, w, t, fps, seed
            )
        if self.transform is not None and not isinstance(self.transform, IdentityTransform):
            video = torch.stack([self.transform(frame) for frame in video], dim=1)

        sample["video"] = video
        return sample


class Kinetics600DataProvider(STAECoreDataProvider):
    def __init__(self, cfg: Kinetics600DataProviderConfig):
        super().__init__(cfg)
        self.cfg: Kinetics600DataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = Kinetics600Dataset(
            self.cfg.data_dir, self.cfg.wds_meta_path, size_transform, transform, self.metadata
        )
        return dataset

    def build_dataset_mask(self, complete_dataset):
        mask = super().build_dataset_mask(complete_dataset)
        mask = mask & np.array(self.metadata["T"] > 0)
        return mask
