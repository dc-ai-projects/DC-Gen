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
import os
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from .base_eval import T2ICoreEvalDataProvider, T2ICoreEvalDataProviderConfig


@dataclass
class MJHQTextPromptDataProviderConfig(T2ICoreEvalDataProviderConfig):
    name: str = "MJHQTextPrompt"
    resolution: int = 512
    num_samples: int = 30000
    data_dir: str = "~/dataset/MJHQ-30K/imgs"
    meta_path: str = "~/dataset/MJHQ-30K/meta_data.json"
    fid_ref_path: str = "~/dataset/MJHQ-30K/MJHQ_30K_${.resolution}px_fid_embeddings_30000.npz"


class MJHQTextPromptDataset(Dataset):
    def __init__(self, json_path: str, num_samples: int = 30000, seed: int = 0):
        super().__init__()
        with open(json_path, "r") as json_file:
            self.samples = list(json.load(json_file).items())

        def cmp(x, y):
            if x[1]["category"] < y[1]["category"]:
                return -1
            elif x[1]["category"] > y[1]["category"]:
                return 1
            else:
                if x[0] < y[0]:
                    return -1
                elif x[0] > y[0]:
                    return 1
                else:
                    return 0

        self.samples.sort(key=cmp_to_key(cmp))

        if num_samples != 30000:
            random_state = np.random.RandomState(seed)
            random_indices = random_state.choice(len(self.samples), size=num_samples, replace=False).tolist()
            self.samples = [self.samples[i] for i in random_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "index": index,
            "name": self.samples[index][0],
            "category": self.samples[index][1]["category"],
            "prompt": self.samples[index][1]["prompt"],
        }
        return sample


class MJHQTextPromptDataProvider(T2ICoreEvalDataProvider):
    def __init__(self, cfg: MJHQTextPromptDataProviderConfig):
        super().__init__(cfg)
        self.cfg: MJHQTextPromptDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = MJHQTextPromptDataset(
            json_path=os.path.expanduser(self.cfg.meta_path),
            num_samples=self.cfg.num_samples,
            seed=self.cfg.seed,
        )
        return dataset
