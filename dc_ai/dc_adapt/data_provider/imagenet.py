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

import os
from dataclasses import dataclass

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder

from ...apps.utils.image import CustomImageFolder, DMCrop
from .base import DCAdaptDataProvider, DCAdaptDataProviderConfig

__all__ = [
    "DCAdaptImageNetLatentProviderConfig",
    "DCAdaptImageNetLatentProvider",
    "DCAdaptImageNetImageProviderConfig",
    "DCAdaptImageNetImageProvider",
]


class _AlignDatasetFolder(Dataset):
    def __init__(self, dataset1: DatasetFolder, dataset2: DatasetFolder) -> None:
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # check datasets
        assert len(self.dataset1) == len(self.dataset2), f"Dataset size mismatch"
        assert self.dataset1.class_to_idx == self.dataset2.class_to_idx, f"Dataset classes mismatch"
        for (path1, class1), (path2, class2) in zip(self.dataset1.samples, self.dataset2.samples):
            assert os.path.basename(path1) == os.path.basename(path2), f"File name mismatch: {path1} <-> {path2}"
            assert class1 == class2, f"Class mismatch: {class1} != {class2}"

    def __len__(self) -> int:
        return len(self.dataset1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data1, label1 = self.dataset1[index]
        data2, label2 = self.dataset2[index]
        assert label1 == label2, f"Class mismatch: {label1} != {label2}"

        data1 = torch.tensor(data1, dtype=torch.float32)
        data2 = torch.tensor(data2, dtype=torch.float32)
        label = torch.tensor(label1, dtype=torch.long)

        return {
            "data1": data1,
            "data2": data2,
            "label": label,
        }


@dataclass
class DCAdaptImageNetLatentProviderConfig(DCAdaptDataProviderConfig):
    name: str = "imagenet"
    drop_last: bool = True
    shuffle: bool = True

    teacher_data_dir: str = "assets/data/latent/sd_vae_ft_ema/imagenet_512"
    student_data_dir: str = "assets/data/latent/dc_ae_f64c128_in_1.0/imagenet_512"


class DCAdaptImageNetLatentProvider(DCAdaptDataProvider):
    def __init__(self, cfg: DCAdaptImageNetLatentProviderConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptImageNetLatentProviderConfig

    def build_dataset(self) -> Dataset:
        dataset1 = DatasetFolder(self.cfg.teacher_data_dir, np.load, (".npy",))
        dataset2 = DatasetFolder(self.cfg.student_data_dir, np.load, (".npy",))
        return _AlignDatasetFolder(dataset1, dataset2)


@dataclass
class DCAdaptImageNetImageProviderConfig(DCAdaptDataProviderConfig):
    name: str = "imagenet"
    drop_last: bool = True
    shuffle: bool = True

    data_dir: str = "~/dataset/imagenet/train"
    resolution: int = 512


class DCAdaptImageNetImageProvider(DCAdaptDataProvider):
    def __init__(self, cfg: DCAdaptImageNetImageProviderConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptImageNetImageProviderConfig

    def build_dataset(self) -> Dataset:
        self.transform = transforms.Compose(
            [
                DMCrop(self.cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5, inplace=True),
            ]
        )
        dataset = CustomImageFolder(self.cfg.data_dir, self.transform, return_dict=True)
        return dataset
