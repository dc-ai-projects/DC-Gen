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
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset

from ...apps.data_provider.dc_base import BaseDataProvider, BaseDataProviderConfig
from ...apps.data_provider.sampler import DistributedRangedAspectRatioBatchSampler, DistributedRangedSampler
from ...apps.data_provider.web_dataset.wids import WebDataset
from ...apps.utils.aspect_ratio import AspectRatioManager512, AspectRatioManager1024, AspectRatioManager2048

__all__ = ["T2ICoreLatentTrainDataProviderConfig", "T2ICoreLatentDataset", "T2ICoreLatentTrainDataProvider"]


@dataclass
class T2ICoreTrainDataProviderConfig(BaseDataProviderConfig):
    resolution: int = 512
    wds_meta_dir: str = MISSING
    wds_meta_filename: str = MISSING
    temperature: float = 0.1
    save_checkpoint_steps: int = MISSING

    shuffle_chunk_size: Optional[int] = 1000
    drop_last: bool = True


@dataclass
class T2ICoreLatentTrainDataProviderConfig(T2ICoreTrainDataProviderConfig):
    data_ext: str = ".npy"


class T2ICoreLatentDataset(WebDataset):
    def __init__(
        self,
        meta_path: str,
        temperature: float,
        data_ext: str,
    ):
        super().__init__(data_dir=None, meta_path=meta_path)
        self.temperature = temperature
        self.data_ext = data_ext

    def __getitem__(self, index: tuple[int, int]) -> dict[str, Any]:
        index, seed = index
        sample = self.dataset[index]
        latent = sample[self.data_ext]

        scores = [float(score) for score in sample[".json"]["clip_scores"]]
        weights = np.array(scores) ** (1.0 / max(self.temperature, 0.01))
        probs = weights / np.sum(weights)

        random_state = np.random.RandomState(seed % 2**32)
        selected_idx = random_state.choice(range(len(sample[".json"]["captions"])), p=probs)
        caption = sample[".json"]["captions"][selected_idx]
        if caption is None:
            caption = ""

        return {
            "images": latent,
            "captions": caption,
            "height": sample[".json"]["height"],
            "width": sample[".json"]["width"],
        }

    def get_data_info(self, index: tuple[int, int]) -> dict[str, Any]:
        index, seed = index
        sample = self.dataset[index]
        return sample[".json"]


class T2ICoreDataProvider(BaseDataProvider):
    def __init__(self, cfg: T2ICoreTrainDataProviderConfig):
        super().__init__(cfg)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask is True:
            return complete_dataset
        else:
            raise ValueError(f"mask {mask} is not supported for T2ICoreDataProvider")


class T2ICoreLatentTrainDataProvider(T2ICoreDataProvider):
    def __init__(self, cfg: T2ICoreLatentTrainDataProviderConfig):
        if cfg.resolution == 512:
            self.aspect_ratio_manager = AspectRatioManager512()
        elif cfg.resolution == 1024:
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == 2048:
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for SanaMSCrop")

        super().__init__(cfg)
        self.cfg: T2ICoreLatentTrainDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = T2ICoreLatentDataset(
            meta_path=os.path.join(self.cfg.wds_meta_dir, self.cfg.wds_meta_filename),
            temperature=self.cfg.temperature,
            data_ext=self.cfg.data_ext,
        )
        return dataset

    def build_sampler(self) -> DistributedRangedAspectRatioBatchSampler:
        raw_sampler = DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
        )
        sampler = DistributedRangedAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=self.cfg.save_checkpoint_steps,
            aspect_ratio_manager=self.aspect_ratio_manager,
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
