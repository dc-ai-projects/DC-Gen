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
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler

from ...apps.data_provider.dc_base import BaseDataProvider, BaseDataProviderConfig
from ...apps.data_provider.sampler import MultiResolutionDistributedRangedSampler
from ...apps.utils.image import DMCrop, DMCropOrUpsampleCrop, MultiResolutionSubset, Resize, UpsampleCrop

__all__ = ["AECoreDataProviderConfig", "AECoreDataProvider"]


@dataclass
class AECoreDataProviderConfig(BaseDataProviderConfig):
    resolution: Any = 256  # int | tuple[int]
    size_transform: str = "DMCrop"
    shuffle_chunk_size: Optional[int] = None
    fid_ref_path: Optional[str] = None
    fvd_ref_path: Optional[str] = None
    mean: float = 0.5
    std: float = 0.5
    vlm_caption_path: Optional[str] = None
    min_h: Optional[int] = None
    min_w: Optional[int] = None


class AECoreDataProvider(BaseDataProvider):
    def __init__(self, cfg: AECoreDataProviderConfig):
        self.vlm_caption = (
            pandas.read_csv(cfg.vlm_caption_path, index_col=0) if cfg.vlm_caption_path is not None else None
        )
        super().__init__(cfg)
        self.cfg: AECoreDataProviderConfig

    def build_transform(self) -> tuple[Callable, Callable]:
        if self.cfg.size_transform == "DMCrop":
            size_transform = DMCrop()
        elif self.cfg.size_transform == "UpsampleCrop":
            size_transform = UpsampleCrop()
        elif self.cfg.size_transform == "DMCropOrUpsampleCrop":
            size_transform = DMCropOrUpsampleCrop()
        elif self.cfg.size_transform == "Resize":
            size_transform = Resize()
        elif self.cfg.size_transform == "ResizeBilinear":
            size_transform = Resize(method="bilinear")
        else:
            raise ValueError(f"size transform {self.cfg.size_transform} is not supported")
        transforms_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.cfg.mean, self.cfg.std),
        ]
        return size_transform, transforms.Compose(transforms_list)

    def build_dataset_mask(self, complete_dataset: Dataset) -> bool | np.ndarray:
        mask = super().build_dataset_mask(complete_dataset)
        # size
        if self.cfg.metadata_path is not None:
            if self.cfg.min_h is not None:
                mask = mask & np.array(self.metadata["H"] >= self.cfg.min_h)
            if self.cfg.min_w is not None:
                mask = mask & np.array(self.metadata["W"] >= self.cfg.min_w)
        else:
            assert (
                self.cfg.min_h is None and self.cfg.min_w is None
            ), f"metadata_path is required to support min_h ({self.cfg.min_h}) and min_w ({self.cfg.min_w})"
        return mask

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask is not True:
            indices = np.where(mask)[0]
            dataset = MultiResolutionSubset(complete_dataset, indices)
        else:
            dataset = complete_dataset
        return dataset

    def build_sampler(self) -> Sampler:
        sampler = MultiResolutionDistributedRangedSampler(
            self.dataset,
            self.cfg.batch_size,
            self.cfg.resolution,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
        )
        return sampler
