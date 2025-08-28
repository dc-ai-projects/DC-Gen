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
from typing import Optional

import numpy as np
import pandas
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from ..utils.dist import get_dist_rank, get_dist_size, is_master

__all__ = ["BaseDataProviderConfig", "BaseDataProvider"]


@dataclass
class BaseDataProviderConfig:
    name: str = MISSING
    batch_size: int = 32
    num_workers: int = 8
    drop_last: bool = False
    seed: int = 0
    shuffle: bool = False
    start_index: int = 0
    end_index: Optional[int] = None
    metadata_path: Optional[str] = None
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = True


class BaseDataProvider:
    def __init__(self, cfg: BaseDataProviderConfig):
        self.cfg = cfg
        self.metadata = pandas.read_csv(cfg.metadata_path, index_col=0) if cfg.metadata_path is not None else None
        self.dataset = self.build_dataset()
        if is_master():
            print(f"len({cfg.name})={len(self.dataset)}")
        self.dist_size = get_dist_size()
        self.rank = get_dist_rank()
        self.sampler = self.build_sampler()
        self.data_loader = self.build_data_loader()

    def build_complete_dataset(self) -> Dataset:
        raise NotImplementedError

    def build_dataset_mask(self, complete_dataset: Dataset) -> bool | np.ndarray:
        mask = True
        if self.cfg.start_index != 0:
            mask = mask & (np.arange(len(complete_dataset)) >= self.cfg.start_index)
        if self.cfg.end_index is not None:
            mask = mask & (np.arange(len(complete_dataset)) < self.cfg.end_index)
        return mask

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        raise NotImplementedError

    def build_dataset(self) -> Dataset:
        dataset = self.build_complete_dataset()
        mask = self.build_dataset_mask(dataset)
        dataset = self.build_filtered_dataset(dataset, mask)
        return dataset

    def build_sampler(self) -> Sampler:
        raise NotImplementedError

    def collate_fn(self, batch):
        return default_collate(batch)

    def build_data_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            sampler=self.sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=self.cfg.drop_last,
            collate_fn=self.collate_fn,
            generator=generator,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def set_batch_index(self, batch_index: int) -> None:
        self.sampler.set_iter_index(batch_index * self.cfg.batch_size)
