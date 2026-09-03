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
import json
import os
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from .base_eval import VideoGenCoreEvalDataProvider, VideoGenCoreEvalDataProviderConfig


@dataclass
class VBench2DataProviderConfig(VideoGenCoreEvalDataProviderConfig):
    name: str = "VBench2"
    meta_path: str = "assets/data/vbench/VBench2_extended_full_info.json"


class VBench2Dataset(Dataset):
    def __init__(self, meta_path: str):
        with open(meta_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        self.samples = []
        for sample in samples:
            repeats = 20 if sample["dimension"][0] == "Diversity" else 3
            for idx in range(repeats):
                new_sample = copy.deepcopy(sample)
                new_sample["video_path"] = os.path.join(sample["dimension"][0], f"{sample['id'][:180]}-{idx}.mp4")
                self.samples.append(new_sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


class VBench2DataProvider(VideoGenCoreEvalDataProvider):
    def __init__(self, cfg: VBench2DataProviderConfig):
        super().__init__(cfg)
        self.cfg: VBench2DataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = VBench2Dataset(meta_path=self.cfg.meta_path)
        return dataset
