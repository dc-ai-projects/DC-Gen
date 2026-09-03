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
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset

from .base_train import T2ICoreLatentTrainDataProvider, T2ICoreLatentTrainDataProviderConfig


@dataclass
class LatentDummyDataProviderConfig(T2ICoreLatentTrainDataProviderConfig):
    name: str = "LatentDummy"
    wds_meta_dir: str = ""
    wds_meta_filename: str = ""
    latent_shape: tuple[int] = MISSING


class LatentDummyDataset(Dataset):
    def __init__(
        self,
        resolution: int,
        latent_shape: tuple[int],
    ):
        self.resolution = resolution
        self.latent_shape = latent_shape

    def __len__(self):
        return 100000000

    def __getitem__(self, index: tuple[int, int]) -> dict[str, Any]:
        index, seed = index
        random_state = np.random.RandomState(seed % 2**32)
        latent = torch.Tensor(random_state.randn(*self.latent_shape))
        return {
            "images": latent,
            "captions": "",
            "height": self.resolution,
            "width": self.resolution,
        }


class LatentDummyDataProvider(T2ICoreLatentTrainDataProvider):
    def __init__(self, cfg: LatentDummyDataProviderConfig):
        super().__init__(cfg)
        self.cfg: LatentDummyDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = LatentDummyDataset(self.cfg.resolution, self.cfg.latent_shape)
        return dataset
