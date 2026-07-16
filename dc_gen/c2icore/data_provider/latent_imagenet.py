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
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder

from .base import C2ICoreDataProvider, C2ICoreDataProviderConfig

__all__ = ["LatentImageNetDataProviderConfig", "LatentImageNetDataProvider"]


@dataclass
class LatentImageNetDataProviderConfig(C2ICoreDataProviderConfig):
    name: str = "LatentImageNet"
    data_dir: str = "assets/data/latent/dc_ae_f32c32/imagenet_512"
    drop_last: bool = True
    shuffle: bool = True


class LatentImageNetDataProvider(C2ICoreDataProvider):
    def __init__(self, cfg: LatentImageNetDataProviderConfig):
        super().__init__(cfg)
        self.cfg: LatentImageNetDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        return DatasetFolder(self.cfg.data_dir, np.load, [".npy"])

    def collate_fn(self, batch: list[np.ndarray]) -> dict[str, torch.Tensor]:
        images, labels = zip(*batch)
        images = torch.from_numpy(np.stack(images))
        labels = torch.tensor(labels)
        return {"images": images, "labels": labels}
