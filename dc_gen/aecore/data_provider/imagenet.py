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

from torch.utils.data import Dataset

from ...apps.utils.image import MultiResolutionImageFolder
from .base import AECoreDataProvider, AECoreDataProviderConfig

__all__ = ["ImageNetDataProviderConfig", "ImageNetDataProvider"]


@dataclass
class ImageNetDataProviderConfig(AECoreDataProviderConfig):
    name: str = "ImageNet"


@dataclass
class ImageNetEvalDataProviderConfig(ImageNetDataProviderConfig):
    name: str = "ImageNetEval"
    resolution: int = 256
    data_dir: str = "~/dataset/imagenet/val"
    metadata_path: str = "assets/data/examination/ImageNet_eval.csv"

    def __post_init__(self):
        self.fid_ref_path: str = f"assets/data/fid/imagenet_eval_{self.resolution}.npz"


@dataclass
class ImageNetTrainDataProviderConfig(ImageNetDataProviderConfig):
    name: str = "ImageNetTrain"
    data_dir: str = "~/dataset/imagenet/train"
    shuffle: bool = True
    drop_last: bool = True
    metadata_path: str = "assets/data/examination/ImageNet_train.csv"


class ImageNetDataProvider(AECoreDataProvider):
    def __init__(self, cfg: ImageNetDataProviderConfig):
        super().__init__(cfg)
        self.cfg: ImageNetDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = MultiResolutionImageFolder(self.cfg.data_dir, size_transform, transform, vlm_caption=self.vlm_caption)
        return dataset
