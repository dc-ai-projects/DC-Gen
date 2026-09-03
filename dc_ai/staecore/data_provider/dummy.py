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
import torch
from PIL import Image
from torch.utils.data import Dataset

from ...apps.utils.video import VideoLoader, parse_index, write_video
from .base import STAECoreDataProvider, STAECoreDataProviderConfig


@dataclass
class DummyDataProviderConfig(STAECoreDataProviderConfig):
    name: str = "Dummy"
    size_transform: str = "TemporalCenterCropSpatialDMCrop"


class DummyDataset(Dataset):
    def __init__(
        self,
        transform: Optional[Callable] = None,
    ):
        self.transform = transform

    def __len__(self):
        return 100000000

    def __getitem__(self, index: int | tuple[int, int, Optional[float], int, int]) -> dict[str, Any]:
        index, h, w, t, fps, seed = parse_index(index)
        random_state = np.random.RandomState(seed % 2**32)
        video = []
        for _ in range(t):
            video.append(Image.fromarray(random_state.randint(low=0, high=256, size=(h, w, 3), dtype=np.uint8)))
        if self.transform is not None:
            video = torch.stack([self.transform(frame) for frame in video], dim=1)

        return {"video": video}


class DummyDataProvider(STAECoreDataProvider):
    def __init__(self, cfg: DummyDataProviderConfig):
        super().__init__(cfg)
        self.cfg: DummyDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = DummyDataset(transform)
        return dataset

    def build_dataset_mask(self, complete_dataset) -> bool | np.ndarray:
        return True
