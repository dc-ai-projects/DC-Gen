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

import copy
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import BatchSampler, Dataset, Sampler

from ..utils.aspect_ratio import BaseAspectRatioManager
from ..utils.dist import get_dist_rank, is_dist_initialized, sync_object, sync_tensor
from .dc_mixture import MixtureSampler

__all__ = ["DistributedRangedSampler"]


class DistributedRangedSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        shuffle_chunk_size: Optional[int] = None,
    ):
        assert rank >= 0 and rank < num_replicas
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        if drop_last:
            self.num_samples_per_rank = self.num_samples // num_replicas
        else:
            self.num_samples_per_rank = (self.num_samples - 1) // num_replicas + 1
        self.shuffle_chunk_size = shuffle_chunk_size
        if shuffle_chunk_size is not None:
            start = self.rank * self.num_samples_per_rank
            end = (self.rank + 1) * self.num_samples_per_rank
            self.ranges = [(i, min(i + shuffle_chunk_size, end)) for i in range(start, end, shuffle_chunk_size)]
        self.epoch = 0
        self.iter_index = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.iter_index = 0

    def set_iter_index(self, iter_index):
        self.iter_index = iter_index

    def __len__(self):
        return self.num_samples_per_rank

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
            if not self.drop_last:
                total_size = self.num_replicas * self.num_samples_per_rank
                padding_size = total_size - len(indices)
                indices += (indices * ((padding_size - 1) // len(indices) + 1))[:padding_size]
            indices = indices[self.rank * self.num_samples_per_rank : (self.rank + 1) * self.num_samples_per_rank]
            assert len(indices) == self.num_samples_per_rank
            yield from indices[self.iter_index :]
        elif self.shuffle_chunk_size is not None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            shard_indices = torch.randperm(len(self.ranges), generator=g).tolist()
            cnt = 0
            for shard_index in shard_indices:
                shard_start, shard_end = self.ranges[shard_index]
                num_shard_samples = shard_end - shard_start
                sample_indices = (
                    (torch.randperm(num_shard_samples, generator=g) + shard_start) % self.num_samples
                ).tolist()
                if cnt + num_shard_samples <= self.iter_index:
                    cnt += num_shard_samples
                    continue
                yield from sample_indices[max(self.iter_index - cnt, 0) :]
                cnt += num_shard_samples
        else:
            start = self.rank * self.num_samples_per_rank + self.iter_index
            end = (self.rank + 1) * self.num_samples_per_rank
            indices = (torch.arange(self.num_replicas * self.num_samples_per_rank) % self.num_samples).tolist()
            yield from indices[start:end]


class MultiResolutionDistributedRangedSampler(DistributedRangedSampler):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        resolution: int | list[int],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        num_samples: Optional[int] = None,
        shuffle_chunk_size: Optional[int] = None,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            num_samples=num_samples,
            shuffle_chunk_size=shuffle_chunk_size,
        )
        self.batch_size = batch_size
        resolution_list = [resolution] if isinstance(resolution, int) else resolution
        self.resolution_list = torch.tensor(resolution_list, dtype=int)

    def __iter__(self):
        indices = list(super().__iter__())
        g = torch.Generator()
        g.manual_seed(self.epoch)
        resolution_indices = torch.randint(
            0, len(self.resolution_list), size=((self.num_samples_per_rank - 1) // self.batch_size + 1,), generator=g
        ).repeat_interleave(self.batch_size)[: self.num_samples_per_rank]
        resolutions = self.resolution_list[resolution_indices].tolist()
        seeds = torch.randint(0, 2**63 - 1, (self.num_samples_per_rank,), generator=g).tolist()
        return iter(zip(indices, resolutions[self.iter_index :], seeds[self.iter_index :]))


class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        save_checkpoint_steps: Optional[int],
        aspect_ratio_manager: BaseAspectRatioManager,
        drop_last: bool = False,
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError(f"sampler should be an instance of ``Sampler``, but got {sampler}")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")

        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_checkpoint_steps = save_checkpoint_steps
        self.aspect_ratio_manager = aspect_ratio_manager
        self.drop_last = drop_last
        self.aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratio_manager.aspect_ratios.keys()}
        self.rank = get_dist_rank()
        self.cur_iters = 0
        self.state_dicts = {}

    def cur_state_dict(self) -> dict:
        aspect_ratio_buckets = copy.deepcopy(sync_object(self.aspect_ratio_buckets))
        state_dict = {
            "cur_iters": self.cur_iters,
            "aspect_ratio_buckets": aspect_ratio_buckets,
        }
        return state_dict

    def reach_save_iters(self):
        self.state_dicts[self.cur_iters] = self.cur_state_dict()

    def state_dict(self, iters: int, remove_previous_iters: bool = True, place_holder: bool = False) -> dict[str, Any]:
        if place_holder:
            return self.cur_state_dict()
        if remove_previous_iters:
            self.state_dicts = {k: v for k, v in self.state_dicts.items() if k >= iters}
        return self.state_dicts[iters]

    def load_state_dict(self, state_dict: dict):
        self.cur_iters = state_dict["cur_iters"]
        self.aspect_ratio_buckets = state_dict["aspect_ratio_buckets"][self.rank]

    def get_data_info(self, index: int | dict) -> Optional[dict]:
        data_info = self.dataset.get_data_info(index)
        if data_info is None:
            return None
        closest_ratio = self.aspect_ratio_manager.get_closest_ratio(data_info["height"], data_info["width"])
        data_info["closest_ratio"] = closest_ratio
        return data_info

    def get_next_index(self, sampler_iter: Iterator[int | dict]) -> Optional[int | dict]:
        try:
            return next(sampler_iter)
        except StopIteration:
            return None

    def __iter__(self) -> Iterator[int | dict]:
        sampler_iter = iter(self.sampler)
        while True:
            index = self.get_next_index(sampler_iter)
            if index is None:
                break
            data_info = self.get_data_info(index)
            if not data_info:
                continue
            bucket = self.aspect_ratio_buckets[data_info["closest_ratio"]]
            bucket.append(index)

            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                indices = copy.deepcopy(bucket)
                bucket.clear()
                self.cur_iters += 1
                if self.save_checkpoint_steps is not None and self.cur_iters % self.save_checkpoint_steps == 0:
                    self.reach_save_iters()
                yield indices

        for bucket in self.aspect_ratio_buckets.values():
            if len(bucket) > 0:
                indices = copy.deepcopy(bucket)
                bucket.clear()
                if not self.drop_last or len(indices) == self.batch_size:
                    self.cur_iters += 1
                    if self.save_checkpoint_steps is not None and self.cur_iters % self.save_checkpoint_steps == 0:
                        self.reach_save_iters()
                    yield indices


class DistributedRangedAspectRatioBatchSampler(AspectRatioBatchSampler):
    def __init__(
        self,
        sampler: DistributedRangedSampler,
        dataset: Dataset,
        batch_size: int,
        save_checkpoint_steps: Optional[int],
        aspect_ratio_manager: BaseAspectRatioManager,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            sampler=sampler,
            dataset=dataset,
            batch_size=batch_size,
            save_checkpoint_steps=save_checkpoint_steps,
            aspect_ratio_manager=aspect_ratio_manager,
            drop_last=drop_last,
        )
        if not isinstance(sampler, DistributedRangedSampler):
            raise TypeError(f"sampler should be an instance of ``DistributedRangedSampler``, but got {sampler}")
        self.sampler: DistributedRangedSampler
        self.sampler_iter_index = 0
        self.async_generator = torch.Generator(device=torch.device("cpu"))
        self.async_generator.manual_seed(self.sampler.seed + self.rank)

    def cur_state_dict(self) -> dict:
        state_dict = super().cur_state_dict()
        state_dict["epoch"] = sync_object(self.sampler.epoch)
        state_dict["iter_index"] = sync_object(self.sampler_iter_index)
        async_generator_state = self.async_generator.get_state()[None]
        if is_dist_initialized():
            async_generator_state = sync_tensor(async_generator_state.cuda(), reduce="cat").cpu()
        state_dict["async_generator_state"] = async_generator_state
        return state_dict

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    def set_iter_index(self, iter_index: int):
        self.sampler_iter_index = iter_index
        self.sampler.set_iter_index(iter_index)

    def load_state_dict(self, state_dict: dict):
        self.sampler.set_epoch(state_dict["epoch"][self.rank])
        self.sampler.set_iter_index(state_dict["iter_index"][self.rank])
        self.async_generator.set_state(state_dict["async_generator_state"][self.rank])
        super().load_state_dict(state_dict)

    def get_next_index(self, sampler_iter: Iterator[int]) -> Optional[tuple[int, int]]:
        try:
            index = next(sampler_iter)
            seed = torch.randint(0, 2**63 - 1, (1,), generator=self.async_generator).item()
            self.sampler_iter_index += 1
            return index, seed
        except StopIteration:
            return None


class MixtureAspectRatioBatchSampler(AspectRatioBatchSampler):
    def __init__(
        self,
        sampler: MixtureSampler,
        dataset: Dataset,
        batch_size: int,
        save_checkpoint_steps: Optional[int],
        aspect_ratio_manager: BaseAspectRatioManager,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            sampler=sampler,
            dataset=dataset,
            batch_size=batch_size,
            save_checkpoint_steps=save_checkpoint_steps,
            aspect_ratio_manager=aspect_ratio_manager,
            drop_last=drop_last,
        )
        if not isinstance(sampler, MixtureSampler):
            raise TypeError(f"sampler should be an instance of ``MixtureSampler``, but got {sampler}")
        self.sampler: MixtureSampler

    def cur_state_dict(self) -> dict:
        state_dict = super().cur_state_dict()
        # rearrange the aspect ratio buckets for FSDP
        for aspect_ratio_buckets_per_rank in state_dict["aspect_ratio_buckets"]:
            for aspect_ratio_key in aspect_ratio_buckets_per_rank:
                # [{"dataset_index": 0, "sample_index": 0, "seed": 0}, {"dataset_index": 0, "sample_index": 1, "seed": 1}, ...] -> {"dataset_index": [0, 0, ...], "sample_index": [0, 1, ...], "seed": [0, 1, ...]}
                aspect_ratio_buckets_per_rank[aspect_ratio_key] = {
                    key: [index[key] for index in aspect_ratio_buckets_per_rank[aspect_ratio_key]]
                    for key in self.sampler.index_keys
                }
        state_dict["sampler_state_dict"] = self.sampler.cur_state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        self.sampler.load_state_dict(state_dict["sampler_state_dict"])
        aspect_ratio_buckets = state_dict["aspect_ratio_buckets"][self.rank]
        # rearrange the aspect ratio buckets back
        for key in aspect_ratio_buckets:
            aspect_ratio_buckets[key] = [
                dict(zip(aspect_ratio_buckets[key].keys(), values))
                for values in zip(*aspect_ratio_buckets[key].values())
            ]
        super().load_state_dict(state_dict)
