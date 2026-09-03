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
from ...apps.utils.dist import is_dist_initialized
from .collection import possible_video_train_data_providers


@dataclass
class STAECoreMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "STAECoreMixture"
    resolution_list: tuple[int] = (256,)
    resolution_sample_ratio: Optional[tuple[float]] = None
    num_frames_list: tuple[int] = (16,)
    num_frames_sample_ratio: Optional[tuple[float]] = None
    fps_list: tuple[Optional[int]] = (None,)
    fps_sample_ratio: Optional[tuple[float]] = None
    size_transform: str = "TemporalRandomCropSpatialDMCrop"
    min_t: int = 64
    min_fps: float = 23
    discard_small_samples: bool = True
    different_num_frames_at_each_rank: bool = False


class STAECoreMixtureDataset(MixtureDataset):
    def __init__(self, cfg: STAECoreMixtureDataProviderConfig, datasets: list[Dataset]):
        super().__init__(cfg, datasets)

    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        dataset_index, sample_index, h, w, t, fps, seed = (
            index["dataset_index"],
            index["sample_index"],
            index["h"],
            index["w"],
            index["t"],
            index["fps"],
            index["seed"],
        )
        assert h == w, f"currently only h == w is supported, but got {h} and {w}"
        dataset_name = self.cfg.data_providers[dataset_index]
        try:
            sample = self.datasets[dataset_index][sample_index, h, w, t, fps, seed]
        except Exception as e:
            print(f"failed to load sample {index['sample_index']} from dataset {dataset_name}")
            raise e
        sample["dataset_name"] = dataset_name
        sample["index"] = index
        return sample


class STAECoreMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: STAECoreMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[MultiResolutionDistributedRangedSampler],
    ):
        super().__init__(cfg, datasets, samplers)
        self.cfg: STAECoreMixtureDataProviderConfig

        resolution_sample_ratio = (
            [1] * len(cfg.resolution_list) if cfg.resolution_sample_ratio is None else cfg.resolution_sample_ratio
        )
        assert len(cfg.resolution_list) == len(resolution_sample_ratio)
        self.resolution_sample_ratio = torch.tensor(resolution_sample_ratio, dtype=torch.float)

        num_frames_sample_ratio = (
            [1] * len(cfg.num_frames_list) if cfg.num_frames_sample_ratio is None else cfg.num_frames_sample_ratio
        )
        assert len(cfg.num_frames_list) == len(num_frames_sample_ratio)
        self.num_frames_sample_ratio = torch.tensor(num_frames_sample_ratio, dtype=torch.float)

        fps_sample_ratio = [1] * len(cfg.fps_list) if cfg.fps_sample_ratio is None else cfg.fps_sample_ratio
        assert len(cfg.fps_list) == len(fps_sample_ratio)
        self.fps_sample_ratio = torch.tensor(fps_sample_ratio, dtype=torch.float)

    def get_index_keys(self) -> set[str]:
        index_keys = super().get_index_keys()
        index_keys.update({"h", "w", "t", "fps"})
        return index_keys

    def generate_resolution_indices(self):
        resolution_indices = torch.multinomial(
            self.resolution_sample_ratio,
            num_samples=self.cfg.save_checkpoint_steps,
            replacement=True,
            generator=self.sync_generator,
        )
        return resolution_indices

    def generate_num_frames_indices(self):
        if self.cfg.different_num_frames_at_each_rank:
            num_frames_indices = torch.multinomial(
                self.num_frames_sample_ratio,
                num_samples=self.cfg.save_checkpoint_steps,
                replacement=True,
                generator=self.async_generator,
            )
        else:
            num_frames_indices = torch.multinomial(
                self.num_frames_sample_ratio,
                num_samples=self.cfg.save_checkpoint_steps,
                replacement=True,
                generator=self.sync_generator,
            )
        return num_frames_indices

    def generate_fps_indices(self):
        fps_indices = torch.multinomial(
            self.fps_sample_ratio,
            num_samples=self.cfg.save_checkpoint_steps,
            replacement=True,
            generator=self.sync_generator,
        )
        return fps_indices

    def reach_save_iters(self):
        super().reach_save_iters()
        self.resolution_indices = self.generate_resolution_indices()
        self.num_frames_indices = self.generate_num_frames_indices()
        self.fps_indices = self.generate_fps_indices()

    def generate_index(self, cur_iters: int) -> dict[str, Any]:
        index = super().generate_index(cur_iters)
        resolution_index = self.resolution_indices[(self.cur_iters % self.save_iters) // self.cfg.batch_size]
        index["h"] = index["w"] = self.cfg.resolution_list[resolution_index]
        num_frames_index = self.num_frames_indices[(self.cur_iters % self.save_iters) // self.cfg.batch_size]
        index["t"] = self.cfg.num_frames_list[num_frames_index]
        fps_index = self.fps_indices[(self.cur_iters % self.save_iters) // self.cfg.batch_size]
        index["fps"] = self.cfg.fps_list[fps_index]
        return index


class STAECoreMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: STAECoreMixtureDataProviderConfig):
        super().__init__(cfg)
        self.cfg: STAECoreMixtureDataProviderConfig

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[MultiResolutionDistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets: list[Dataset] = []
        samplers: list[MultiResolutionDistributedRangedSampler] = []
        max_resolution = max(self.cfg.resolution_list)
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name in possible_video_train_data_providers:
                seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
                data_provider_cfg = possible_video_train_data_providers[data_provider_name][0](
                    seed=seed,
                    size_transform=self.cfg.size_transform,
                    min_h=max_resolution if self.cfg.discard_small_samples else None,
                    min_w=max_resolution if self.cfg.discard_small_samples else None,
                    min_t=self.cfg.min_t,
                    min_fps=self.cfg.min_fps,
                )
                data_provider = possible_video_train_data_providers[data_provider_name][1](data_provider_cfg)
            else:
                raise ValueError(f"data provider {data_provider_name} is not supported in mixture data provider")
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> STAECoreMixtureDataset:
        return STAECoreMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise NotImplementedError

    def build_sampler(self) -> STAECoreMixtureSampler:
        return STAECoreMixtureSampler(self.cfg, self.datasets, self.samplers)

    def collate_fn(self, batch):
        collated_batch = {}
        collated_batch["videos"] = default_collate([item.pop("video") for item in batch])
        if "latent" in batch[0]:
            collated_batch["latents"] = default_collate([item.pop("latent") for item in batch])
        keys = set()
        for item in batch:
            keys.update(item.keys())
        collated_batch.update({key: [item.get(key, None) for item in batch] for key in keys})
        return collated_batch
