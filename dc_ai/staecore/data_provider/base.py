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
from typing import Callable, Optional

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler

from ...aecore.data_provider.base import AECoreDataProvider, AECoreDataProviderConfig
from ...apps.data_provider.sampler import DistributedRangedSampler
from ...apps.utils.video import MultiResolutionVideoSubset, VideoSizeTransform


@dataclass
class STAECoreDataProviderConfig(AECoreDataProviderConfig):
    h: Optional[int] = None
    w: Optional[int] = None
    t: Optional[int] = None
    fps: Optional[float] = None

    min_t: Optional[int] = None
    min_fps: Optional[float] = None


class STAECoreDataProvider(AECoreDataProvider):
    def __init__(self, cfg: STAECoreDataProviderConfig):
        super().__init__(cfg)
        self.cfg: STAECoreDataProviderConfig

    def build_transform(self) -> tuple[Callable, Callable]:
        if self.cfg.size_transform == "TemporalCenterCropSpatialResizeResampleFloor":
            size_transform = VideoSizeTransform(
                temporal_transform="CenterCrop",
                spatial_transform="Resize",
                resample_method="Floor",
                h=self.cfg.h,
                w=self.cfg.w,
                t=self.cfg.t,
                fps=self.cfg.fps,
            )
        elif self.cfg.size_transform == "TemporalRandomCropSpatialResizeCenterCrop":
            size_transform = VideoSizeTransform(
                temporal_transform="RandomCrop",
                spatial_transform="ResizeCenterCrop",
                resample_method=None,
                h=self.cfg.h,
                w=self.cfg.w,
                t=self.cfg.t,
                fps=self.cfg.fps,
            )
        elif self.cfg.size_transform == "TemporalRandomCropSpatialDMCrop":
            size_transform = VideoSizeTransform(
                temporal_transform="RandomCrop",
                spatial_transform="DMCrop",
                resample_method=None,
                h=self.cfg.h,
                w=self.cfg.w,
                t=self.cfg.t,
                fps=self.cfg.fps,
            )
        elif self.cfg.size_transform == "TemporalCenterCropSpatialDMCrop":
            size_transform = VideoSizeTransform(
                temporal_transform="CenterCrop",
                spatial_transform="DMCrop",
                resample_method=None,
                h=self.cfg.h,
                w=self.cfg.w,
                t=self.cfg.t,
                fps=self.cfg.fps,
            )
        elif self.cfg.size_transform == "TemporalFrontCropSpatialDMCrop":
            size_transform = VideoSizeTransform(
                temporal_transform="FrontCrop",
                spatial_transform="DMCrop",
                resample_method=None,
                h=self.cfg.h,
                w=self.cfg.w,
                t=self.cfg.t,
                fps=self.cfg.fps,
            )
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
            if self.cfg.min_t is not None:
                mask = mask & np.array(self.metadata["T"] >= self.cfg.min_t)
            if self.cfg.min_fps is not None:
                mask = mask & np.array(self.metadata["fps"] >= self.cfg.min_fps)
        else:
            assert (
                self.cfg.min_t is None and self.cfg.min_fps is None
            ), f"metadata_path is required to support min_t ({self.cfg.min_t}) and min_fps ({self.cfg.min_fps})"
        return mask

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask is not True:
            indices = np.where(mask)[0]
            dataset = MultiResolutionVideoSubset(complete_dataset, indices)
        else:
            dataset = complete_dataset
        return dataset

    def build_sampler(self) -> Sampler:
        sampler = DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
        )
        return sampler
