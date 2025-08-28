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
from torch.utils.data import Dataset, default_collate

from ...apps.data_provider.dc_mixture import (
    MixtureDataProvider,
    MixtureDataProviderConfig,
    MixtureDataset,
    MixtureSampler,
)
from ...apps.data_provider.sampler import MultiResolutionDistributedRangedSampler
from ...apps.utils.dist import get_dist_rank
from .collection import possible_train_data_providers


@dataclass
class AECoreMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "AECoreMixture"
    resolution_list: tuple[int] = (256,)
    resolution_sample_ratio: Optional[tuple[float]] = None
    size_transform: str = "DMCrop"
    discard_small_samples: bool = True


class AECoreMixtureDataset(MixtureDataset):
    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        image, label = self.datasets[index["dataset_index"]][index["sample_index"], index["resolution"], index["seed"]]
        dataset_name = self.cfg.data_providers[index["dataset_index"]]
        return {"image": image, "dataset_name": dataset_name, "label": label, "index": index}


class AECoreMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: AECoreMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[MultiResolutionDistributedRangedSampler],
    ):
        super().__init__(cfg, datasets, samplers)
        self.cfg: AECoreMixtureDataProviderConfig
        resolution_sample_ratio = (
            [1] * len(self.cfg.resolution_list)
            if self.cfg.resolution_sample_ratio is None
            else self.cfg.resolution_sample_ratio
        )
        assert len(cfg.resolution_list) == len(resolution_sample_ratio)
        self.resolution_sample_ratio = torch.tensor(resolution_sample_ratio, dtype=torch.float)

    def get_index_keys(self) -> set[str]:
        index_keys = super().get_index_keys()
        index_keys.update({"resolution"})
        return index_keys

    def generate_resolution_indices(self):
        resolution_indices = torch.multinomial(
            self.resolution_sample_ratio,
            num_samples=self.cfg.save_checkpoint_steps,
            replacement=True,
            generator=self.sync_generator,
        )
        return resolution_indices

    def reach_save_iters(self):
        super().reach_save_iters()
        self.resolution_indices = self.generate_resolution_indices()

    def generate_index(self, cur_iters: int) -> dict[str, Any]:
        index = super().generate_index(cur_iters)
        resolution_index = self.resolution_indices[(self.cur_iters % self.save_iters) // self.cfg.batch_size]
        index["resolution"] = self.cfg.resolution_list[resolution_index]
        return index


class AECoreMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: AECoreMixtureDataProviderConfig):
        super().__init__(cfg)
        self.cfg: AECoreMixtureDataProviderConfig

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[MultiResolutionDistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets: list[Dataset] = []
        samplers: list[MultiResolutionDistributedRangedSampler] = []
        max_resolution = max(self.cfg.resolution_list)
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name in possible_train_data_providers:
                seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
                data_provider_cfg = possible_train_data_providers[data_provider_name][0](
                    seed=seed,
                    size_transform=self.cfg.size_transform,
                    min_h=max_resolution if self.cfg.discard_small_samples else None,
                    min_w=max_resolution if self.cfg.discard_small_samples else None,
                )
                data_provider = possible_train_data_providers[data_provider_name][1](data_provider_cfg)
            else:
                raise ValueError(f"data provider {data_provider_name} is not supported in mixture data provider")
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> AECoreMixtureDataset:
        return AECoreMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise NotImplementedError

    def build_sampler(self) -> AECoreMixtureSampler:
        return AECoreMixtureSampler(self.cfg, self.datasets, self.samplers)

    def collate_fn(self, batch):
        images = default_collate([item["image"] for item in batch])
        batch = {key: [item[key] for item in batch] for key in batch[0] if key != "image"}
        batch["images"] = images
        return batch
