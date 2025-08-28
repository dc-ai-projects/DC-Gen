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
from .base_train import T2ICoreLatentTrainDataProviderConfig
from .collection import possible_train_data_providers
from .latent_dummy import LatentDummyDataProvider, LatentDummyDataProviderConfig


@dataclass
class T2ICoreLatentMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "T2ICoreLatentMixture"
    cache_train_states: bool = False
    resolution: int = 512
    shuffle_chunk_size: Optional[int] = 1000
    wds_meta_dir: Optional[str] = None
    latent_shape: Optional[tuple[int]] = None  # for dummy data provider
    data_ext: str = ".npy"


class T2ICoreLatentMixtureDataset(MixtureDataset):
    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        sample = self.datasets[index["dataset_index"]][index["sample_index"], index["seed"]]  # No need for resolution
        sample.update({"dataset_name": self.cfg.data_providers[index["dataset_index"]], "index": index})
        return sample

    def get_data_info(self, index: dict[str, Any]):
        sample = self.__getitem__(index)
        return {
            "height": sample["height"],
            "width": sample["width"],
        }


class T2ICoreLatentMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: T2ICoreLatentMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[DistributedRangedSampler],
    ):
        super().__init__(cfg, datasets, samplers)
        self.cfg: T2ICoreLatentMixtureDataProviderConfig


class T2ICoreLatentMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: T2ICoreLatentMixtureDataProviderConfig):
        if cfg.resolution == 512:
            self.aspect_ratio_manager = AspectRatioManager512()
        elif cfg.resolution == 1024:
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == 2048:
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for SanaMSCrop")

        super().__init__(cfg)
        self.cfg: T2ICoreLatentMixtureDataProviderConfig
        self.sampler: MixtureAspectRatioBatchSampler

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[DistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets: list[Dataset] = []
        samplers: list[DistributedRangedSampler] = []
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name in possible_train_data_providers:
                seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
                data_provider_cfg = possible_train_data_providers[data_provider_name][0](
                    wds_meta_dir=self.cfg.wds_meta_dir,
                    seed=seed,
                    shuffle_chunk_size=self.cfg.shuffle_chunk_size,
                )
                if isinstance(data_provider_cfg, T2ICoreLatentTrainDataProviderConfig):
                    data_provider_cfg.data_ext = self.cfg.data_ext
                data_provider = possible_train_data_providers[data_provider_name][1](data_provider_cfg)
            elif data_provider_name == "LatentDummy":
                assert self.cfg.latent_shape is not None
                data_provider_cfg = LatentDummyDataProviderConfig(
                    resolution=self.cfg.resolution,
                    latent_shape=self.cfg.latent_shape,
                )
                data_provider = LatentDummyDataProvider(data_provider_cfg)
            else:
                raise ValueError(f"data provider {data_provider_name} is not supported in mixture data provider")
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> T2ICoreLatentMixtureDataset:
        return T2ICoreLatentMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise ValueError(f"mask {mask} is not supported for T2ICoreLatentMixtureDataProvider")

    def build_sampler(self) -> MixtureAspectRatioBatchSampler:
        raw_sampler = T2ICoreLatentMixtureSampler(self.cfg, self.datasets, self.samplers)
        sampler = MixtureAspectRatioBatchSampler(
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
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            generator=generator,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )
        return data_loader

    def collate_fn(self, batch):
        images = default_collate([item["images"] for item in batch])
        batch = {key: [item[key] for item in batch] for key in batch[0] if key != "images"}
        batch["images"] = images
        return batch
