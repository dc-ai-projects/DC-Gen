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

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate

from ...apps.data_provider.dc_mixture import (
    MixtureDataProvider,
    MixtureDataProviderConfig,
    MixtureDataset,
    MixtureSampler,
)
from ...apps.data_provider.sampler import DistributedRangedSampler, MixtureAspectRatioBatchSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512,
    AspectRatioManager512F32MS,
    AspectRatioManager512F64MS,
    AspectRatioManager1024,
    AspectRatioManager2048,
    AspectRatioManagerVideo480F32MS,
)
from .collection import possible_train_data_providers


@dataclass
class DCAdaptLatentMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "DCAdaptLatentMixture"
    resolution: str = "512"
    shuffle_chunk_size: Optional[int] = 1000
    teacher_wds_meta_dir: Optional[str] = None
    student_wds_meta_dir: Optional[str] = None
    data_ext: str = ".npy"


@dataclass
class DCAdaptLatentImageMixtureDataProviderConfig(DCAdaptLatentMixtureDataProviderConfig):
    name: str = "DCAdaptLatentImageMixture"
    data_providers: tuple[str, ...] = ()


@dataclass
class DCAdaptLatentImageEditMixtureDataProviderConfig(DCAdaptLatentMixtureDataProviderConfig):
    name: str = "DCAdaptLatentImageEditMixture"
    data_providers: tuple[str, ...] = ("LatentPicoBananaAlign",)


@dataclass
class DCAdaptLatentVideoMixtureDataProviderConfig(DCAdaptLatentMixtureDataProviderConfig):
    name: str = "DCAdaptLatentVideoMixture"
    data_providers: tuple[str, ...] = ("LatentFusionXAlign",)


class DCAdaptLatentMixtureDataset(MixtureDataset):
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


class DCAdaptLatentMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: DCAdaptLatentMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[DistributedRangedSampler],
    ):
        super().__init__(cfg, datasets, samplers)
        self.cfg: DCAdaptLatentMixtureDataProviderConfig
        self.datasets_len = sum(len(dataset) for dataset in datasets)

    def __len__(self):
        return self.datasets_len


class DCAdaptLatentMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: DCAdaptLatentMixtureDataProviderConfig):
        if cfg.resolution == "480F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F32MS()
        elif cfg.resolution == "512":
            self.aspect_ratio_manager = AspectRatioManager512()
        elif cfg.resolution == "512F32MS":
            self.aspect_ratio_manager = AspectRatioManager512F32MS()
        elif cfg.resolution == "512F64MS":
            self.aspect_ratio_manager = AspectRatioManager512F64MS()
        elif cfg.resolution == "1024":
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == "2048":
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for SanaMSCrop")

        super().__init__(cfg)
        self.cfg: DCAdaptLatentMixtureDataProviderConfig
        self.sampler: MixtureAspectRatioBatchSampler

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[DistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets: list[Dataset] = []
        samplers: list[DistributedRangedSampler] = []
        assert self.cfg.teacher_wds_meta_dir is not None and self.cfg.student_wds_meta_dir is not None
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name in possible_train_data_providers:
                seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
                data_provider_cfg = possible_train_data_providers[data_provider_name][0](
                    resolution=self.cfg.resolution,
                    teacher_wds_meta_dir=self.cfg.teacher_wds_meta_dir,
                    student_wds_meta_dir=self.cfg.student_wds_meta_dir,
                    seed=seed,
                    shuffle_chunk_size=self.cfg.shuffle_chunk_size,
                    data_ext=self.cfg.data_ext,
                )
                data_provider = possible_train_data_providers[data_provider_name][1](data_provider_cfg)
            else:
                raise ValueError(f"data provider {data_provider_name} is not supported in mixture data provider")
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> DCAdaptLatentMixtureDataset:
        return DCAdaptLatentMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        return complete_dataset

    def build_sampler(self) -> MixtureAspectRatioBatchSampler:
        raw_sampler = DCAdaptLatentMixtureSampler(self.cfg, self.datasets, self.samplers)
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
        latents1 = default_collate([item["data1"] for item in batch])
        latents2 = default_collate([item["data2"] for item in batch])
        captions = default_collate([item["caption"] for item in batch])

        if "ae_feature1" in batch[0]:
            ae_feature1 = default_collate([item["ae_feature1"] for item in batch])
            ae_feature2 = default_collate([item["ae_feature2"] for item in batch])
        else:
            ae_feature1 = ae_feature2 = None

        batch = {}
        batch["data1"] = latents1
        batch["data2"] = latents2
        batch["caption"] = captions

        if ae_feature1 is not None:
            batch["ae_feature1"] = ae_feature1
            batch["ae_feature2"] = ae_feature2

        return batch
