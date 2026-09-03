# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Mixture provider for T2I image training."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from ...apps.data_provider.dc_mixture import (
    MixtureDataProvider,
    MixtureDataProviderConfig,
    MixtureDataset,
    MixtureSampler,
)
from ...apps.data_provider.sampler import DistributedRangedSampler, MixtureAspectRatioBatchSampler
from ...apps.utils.aspect_ratio import AspectRatioManager512, AspectRatioManager1024, AspectRatioManager2048
from .collection import possible_train_data_providers
from .image import T2ICoreImageTrainDataProviderConfig

__all__ = ["T2ICoreImageMixtureDataProvider", "T2ICoreImageMixtureDataProviderConfig"]


@dataclass
class T2ICoreImageMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "T2ICoreImageMixture"
    cache_train_states: bool = False
    resolution: int = 512
    shuffle_chunk_size: Optional[int] = 1000


class T2ICoreImageMixtureDataset(MixtureDataset):
    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        sample = self.datasets[index["dataset_index"]][index["sample_index"], index["seed"]]
        sample.update({"dataset_name": self.cfg.data_providers[index["dataset_index"]], "index": index})
        return sample

    def get_data_info(self, index: dict[str, Any]) -> Optional[dict[str, int]]:
        return self.datasets[index["dataset_index"]].get_data_info((index["sample_index"], index["seed"]))


class T2ICoreImageMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: T2ICoreImageMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[DistributedRangedSampler],
    ) -> None:
        super().__init__(cfg, datasets, samplers)
        self.datasets_len = sum(len(dataset) for dataset in datasets)

    def __len__(self) -> int:
        return self.datasets_len


class T2ICoreImageMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: T2ICoreImageMixtureDataProviderConfig):
        if cfg.resolution == 512:
            self.aspect_ratio_manager = AspectRatioManager512()
        elif cfg.resolution == 1024:
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == 2048:
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for T2I image mixture")
        super().__init__(cfg)
        self.cfg: T2ICoreImageMixtureDataProviderConfig
        self.sampler: MixtureAspectRatioBatchSampler

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[DistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets = []
        samplers = []
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name not in possible_train_data_providers:
                raise ValueError(f"data provider {data_provider_name} is not supported in image mixture")
            config_class, provider_class = possible_train_data_providers[data_provider_name]
            seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
            data_provider_cfg = config_class(
                resolution=self.cfg.resolution,
                seed=seed,
                shuffle_chunk_size=self.cfg.shuffle_chunk_size,
            )
            if not isinstance(data_provider_cfg, T2ICoreImageTrainDataProviderConfig):
                raise ValueError(f"data provider {data_provider_name} is not an image data provider")
            data_provider = provider_class(data_provider_cfg)
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> T2ICoreImageMixtureDataset:
        return T2ICoreImageMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask is True:
            return complete_dataset
        raise ValueError(f"mask {mask} is not supported for T2ICoreImageMixtureDataProvider")

    def build_sampler(self) -> MixtureAspectRatioBatchSampler:
        raw_sampler = T2ICoreImageMixtureSampler(self.cfg, self.datasets, self.samplers)
        return MixtureAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=self.cfg.save_checkpoint_steps,
            aspect_ratio_manager=self.aspect_ratio_manager,
            drop_last=False,
        )

    def build_data_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            generator=generator,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        images = default_collate([item["image"] for item in batch])
        collated = {key: [item[key] for item in batch] for key in batch[0] if key != "image"}
        collated["image"] = images
        return collated
