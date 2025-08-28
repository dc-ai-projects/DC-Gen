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

import json
import warnings
from functools import lru_cache

import numpy as np
from torch.utils.data import Dataset, get_worker_info

from .single_zip_dataset import SingleZipDataset

_worker_local_caches = {}


def get_zip_dataset_per_worker(path: str):
    """
    Create or retrieve a cached instance of SingleZipDataset for every worker, avoiding file pointer curruption.
    """
    worker_info = get_worker_info()
    worker_id = worker_info.id if worker_info is not None else -1  # main process has id -1

    if worker_id not in _worker_local_caches:
        # create a new cache for every worker
        @lru_cache(maxsize=16)
        def _cached_get_dataset(meta_path: str):
            return SingleZipDataset(meta_path)

        _worker_local_caches[worker_id] = _cached_get_dataset

    return _worker_local_caches[worker_id](path)


class ZipDataset(Dataset):
    def __init__(self, meta_path: str):
        super().__init__()
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            self.shardlist = meta_data["shardlist"]
        self.lengths = [shard["nsamples"] for shard in self.shardlist]
        self.cum_lengths = np.cumsum(self.lengths)
        self.total_length = self.cum_lengths[-1]

    @staticmethod
    def get_zip_dataset(path: str):
        return get_zip_dataset_per_worker(path)

    def get_shard(self, index):
        """Get the shard and index within the shard corresponding to the given index."""
        # Find the shard corresponding to the given index.
        shard_idx = np.searchsorted(self.cum_lengths, index, side="right")

        # Figure out which index within the shard corresponds to the given index.
        if shard_idx == 0:
            inner_idx = index
        else:
            inner_idx = index - self.cum_lengths[shard_idx - 1]

        # Get the shard and return the corresponding element.
        shard = self.shardlist[shard_idx]
        url = shard["url"]
        shard_dataset = self.get_zip_dataset(url)

        # Check if the cache hit rate is critically low.
        self._check_cache_hit_rate()

        return shard_dataset, inner_idx, url

    def __getitem__(self, index):
        shard_dataset, inner_idx, url = self.get_shard(index)
        sample = shard_dataset[inner_idx]

        sample["__index__"] = index
        sample["__shard__"] = url
        sample["__shardindex__"] = inner_idx

        return sample

    def __len__(self):
        return self.total_length

    def _check_cache_hit_rate(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1

        if worker_id in _worker_local_caches:
            cache_func = _worker_local_caches[worker_id]
            info = cache_func.cache_info()
        hits = info.hits
        misses = info.misses
        total_calls = hits + misses

        if total_calls > 100 and total_calls > 0:
            hit_rate = hits / total_calls
            if hit_rate < 0.1:
                warnings.warn(
                    f"Cache hit rate for 'get_zip_dataset' is critically low: {hit_rate:.2%}. "
                    f"Consider increasing cache size (current: {info.maxsize}) or "
                    f"improving data access patterns. (Hits: {hits}, Misses: {misses})",
                    UserWarning,
                )
