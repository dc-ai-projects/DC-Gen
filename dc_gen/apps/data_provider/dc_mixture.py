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
from typing import Any, Iterator

import torch
from omegaconf import MISSING
from torch.utils.data import Dataset, Sampler

from ..utils.dist import get_dist_rank, is_dist_initialized, sync_tensor
from .dc_base import BaseDataProvider, BaseDataProviderConfig


@dataclass
class MixtureDataProviderConfig(BaseDataProviderConfig):
    name: str = "Mixture"
    data_providers: tuple[str, ...] = ()
    data_provider_sample_ratio: Any = None  # Optional[tuple[float]]
    save_checkpoint_steps: int = MISSING
    cache_train_states: bool = True


class MixtureDataset(Dataset):
    def __init__(self, cfg: MixtureDataProviderConfig, datasets: list[Dataset]):
        self.cfg = cfg
        assert len(cfg.data_providers) == len(datasets)
        self.datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class MixtureSampler(Sampler):
    def __init__(
        self,
        cfg: MixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[Sampler],
    ):
        self.cfg = cfg
        self.num_data_providers = len(cfg.data_providers)
        data_provider_sample_ratio = (
            [len(dataset) for dataset in datasets]
            if self.cfg.data_provider_sample_ratio is None
            else self.cfg.data_provider_sample_ratio
        )
        assert self.num_data_providers == len(datasets)
        assert self.num_data_providers == len(samplers)
        assert self.num_data_providers == len(data_provider_sample_ratio)
        self.samplers = samplers
        self.data_provider_sample_ratio = torch.tensor(data_provider_sample_ratio, dtype=torch.float)
        self.rank = get_dist_rank()
        self.sync_generator = torch.Generator(device=torch.device("cpu"))
        self.sync_generator.manual_seed(cfg.seed)
        self.async_generator = torch.Generator(device=torch.device("cpu"))
        self.async_generator.manual_seed(cfg.seed + self.rank)
        self.save_iters = cfg.save_checkpoint_steps * cfg.batch_size
        self.samplers_state = torch.zeros((self.num_data_providers, 2), dtype=int)  # epoch, index
        self.cur_iters = 0
        self.state_dicts = {}
        self.index_keys = self.get_index_keys()

    def get_index_keys(self) -> set[str]:
        return {"dataset_index", "sample_index", "seed"}

    def cur_state_dict(self) -> dict[str, Any]:
        sync_generator_state = self.sync_generator.get_state()[None]
        async_generator_state = self.async_generator.get_state()[None]
        samplers_state = self.samplers_state[None].clone()
        if is_dist_initialized():
            sync_generator_state = sync_tensor(sync_generator_state.cuda(), reduce="cat").cpu()
            async_generator_state = sync_tensor(async_generator_state.cuda(), reduce="cat").cpu()
            samplers_state = sync_tensor(samplers_state.cuda(), reduce="cat").cpu()
        return {
            "cur_iters": self.cur_iters,
            "sync_generator_state": sync_generator_state,
            "async_generator_state": async_generator_state,
            "samplers_state": samplers_state,
        }

    def state_dict(self, iters: int, remove_previous_iters: bool = True, place_holder: bool = False) -> dict[str, Any]:
        if place_holder:
            return self.cur_state_dict()
        if remove_previous_iters:
            self.state_dicts = {k: v for k, v in self.state_dicts.items() if k >= iters}
        return self.state_dicts[iters]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.cur_iters = state_dict["cur_iters"]
        self.sync_generator.set_state(state_dict["sync_generator_state"][self.rank])
        self.async_generator.set_state(state_dict["async_generator_state"][self.rank])
        self.samplers_state = state_dict["samplers_state"][self.rank]

    def reach_save_iters(self):
        if self.cfg.cache_train_states:
            self.state_dicts[self.cur_iters] = self.cur_state_dict()

    def generate_index(self, cur_iters: int) -> dict[str, Any]:
        index = {}
        return index

    def __iter__(self):
        iters: list[Iterator] = []
        for dataset_index, sampler in enumerate(self.samplers):
            sampler.set_epoch(self.samplers_state[dataset_index, 0].item())
            sampler.set_iter_index(self.samplers_state[dataset_index, 1].item())
            iters.append(iter(sampler))

        self.reach_save_iters()
        while True:
            dataset_index = torch.multinomial(self.data_provider_sample_ratio, 1, generator=self.async_generator).item()
            seed = torch.randint(0, 2**63 - 1, (1,), generator=self.async_generator).item()
            try:
                sample_index = next(iters[dataset_index])
                if isinstance(sample_index, tuple):
                    sample_index = sample_index[0]
                assert isinstance(sample_index, int)
            except StopIteration:
                self.samplers_state[dataset_index, 0] += 1
                self.samplers_state[dataset_index, 1] = 0
                self.samplers[dataset_index].set_epoch(self.samplers_state[dataset_index, 0].item())
                self.samplers[dataset_index].set_iter_index(0)
                print(
                    f"rank {self.rank}: dataset {self.cfg.data_providers[dataset_index]} starts epoch {self.samplers_state[dataset_index, 0].item()}"
                )
                iters[dataset_index] = iter(self.samplers[dataset_index])
                sample_index = next(iters[dataset_index])
                if isinstance(sample_index, tuple):
                    sample_index = sample_index[0]
                assert isinstance(sample_index, int)
            index = self.generate_index(self.cur_iters)
            index["dataset_index"] = dataset_index
            index["sample_index"] = sample_index
            index["seed"] = seed
            if set(index.keys()) != self.index_keys:
                raise ValueError(f"index keys {index.keys()} is different from expected keys {self.index_keys}")
            self.samplers_state[dataset_index, 1] += 1
            self.cur_iters += 1
            if self.cur_iters % self.save_iters == 0:
                self.reach_save_iters()
            yield index


class MixtureDataProvider(BaseDataProvider):
    def __init__(self, cfg: MixtureDataProviderConfig):
        self.cfg = cfg
        self.datasets, self.samplers = self.build_datasets_and_samplers()
        super().__init__(cfg)

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[Sampler]]:
        raise NotImplementedError

    def build_complete_dataset(self) -> MixtureDataset:
        raise NotImplementedError  # format: return MixtureDataset(self.cfg, self.datasets)

    def build_sampler(self) -> Sampler:
        raise NotImplementedError  # format: return MixtureSampler(self.cfg, self.datasets, self.samplers)
