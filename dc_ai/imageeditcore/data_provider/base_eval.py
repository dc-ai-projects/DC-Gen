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
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ...apps.data_provider.dc_base import BaseDataProvider, BaseDataProviderConfig
from ...apps.data_provider.sampler import DistributedRangedAspectRatioBatchSampler, DistributedRangedSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512F32MS,
    AspectRatioManager512F64MS,
    AspectRatioManager1024,
    AspectRatioManager2048,
)

__all__ = ["ImageEditCoreEvalDataProviderConfig", "ImageEditCoreEvalDataProvider"]


@dataclass
class ImageEditCoreEvalDataProviderConfig(BaseDataProviderConfig):
    resolution: str = "512F32MS"
    fid_ref_path: Optional[str] = None
    size_transform: str = "QwenImageResize"


class ImageEditCoreEvalDataProvider(BaseDataProvider):
    def __init__(self, cfg: ImageEditCoreEvalDataProviderConfig):
        if cfg.resolution == "512F32MS":
            self.aspect_ratio_manager = AspectRatioManager512F32MS()
        elif cfg.resolution == "512F64MS":
            self.aspect_ratio_manager = AspectRatioManager512F64MS()
        elif cfg.resolution == "1024":
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == "2048":
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for GEdit.")

        super().__init__(cfg)
        self.cfg: ImageEditCoreEvalDataProviderConfig

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask is not True:
            indices = np.where(mask)[0]
            dataset = Subset(complete_dataset, indices)
        else:
            dataset = complete_dataset
        return dataset

    def build_sampler(self) -> DistributedRangedAspectRatioBatchSampler:
        raw_sampler = DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
        )
        sampler = DistributedRangedAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            aspect_ratio_manager=self.aspect_ratio_manager,
            save_checkpoint_steps=100000000,  # No need to save
            drop_last=False,
        )
        return sampler

    def build_data_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            collate_fn=self.collate_fn,
            generator=generator,
        )
        return data_loader
