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
from typing import Any, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ...apps.utils.image import QwenImageResize
from .base_eval import ImageEditCoreEvalDataProvider, ImageEditCoreEvalDataProviderConfig


@dataclass
class GEditDataProviderConfig(ImageEditCoreEvalDataProviderConfig):
    name: str = "GEdit"
    resolution: str = "512F32MS"  # 512F32MS, 512F64MS, 1024, 2048
    num_samples: int = 1212
    data_dir: str = "assets/data/GEdit/images"
    meta_path: str = "assets/data/GEdit/meta_data.json"
    size_transform: str = "QwenImageResize"
    language: str = "all"  # cn, en
    task_type: Optional[str] = None


class GEditDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        json_path: str,
        num_samples: int = 1212,
        seed: int = 0,
        size_transform: str = "QwenImageResize",
        total_compression_ratio: int = 32,
        resolution: int = 512,
        language: str = "all",
        task_type: Optional[str] = None,
    ):
        super().__init__()

        self.samples = []

        with open(json_path, "r") as json_file:
            meta = json.load(json_file)
            for key, value in meta.items():
                for idx, prompt in enumerate(value["prompts"]):
                    if (language == "all" or value["language"] == language) and (
                        task_type is None or value["task_type"] == task_type
                    ):
                        if "path" in value:  # Support path != key
                            img = Image.open(os.path.join(image_dir, value["path"]))
                        else:
                            img = Image.open(os.path.join(image_dir, key + ".png"))

                        self.samples.append(
                            {
                                "image": img,
                                "prompt": prompt,
                                "language": value["language"],
                                "task_type": value["task_type"],
                                "height": value["height"],
                                "width": value["width"],
                                "name": key + f"_{idx}",
                            }
                        )

        def cmp(x, y):
            if x["name"] < y["name"]:
                return -1
            elif x["name"] > y["name"]:
                return 1
            else:
                return 0

        self.samples.sort(key=cmp_to_key(cmp))

        if num_samples != 1212:
            random_state = np.random.RandomState(seed)
            random_indices = random_state.choice(len(self.samples), size=num_samples, replace=False).tolist()
            self.samples = [self.samples[i] for i in random_indices]

        if size_transform == "QwenImageResize":
            self.size_transform = QwenImageResize(
                resolution=resolution, total_compression_ratio=total_compression_ratio
            )
        else:
            raise ValueError(f"Size transform {size_transform} is not supported.")

        self.total_compression_ratio = total_compression_ratio

    def __len__(self) -> int:
        return len(self.samples)

    def get_data_info(self, index: int | tuple) -> dict[str, Any]:
        if isinstance(index, tuple):
            index = index[0]
        sample = {
            "index": index,
            "height": self.samples[index]["height"],
            "width": self.samples[index]["width"],
            "prompt": self.samples[index]["prompt"],
            "name": self.samples[index]["name"],
            "language": self.samples[index]["language"],
            "task_type": self.samples[index]["task_type"],
        }
        return sample

    def __getitem__(self, index: int | tuple) -> dict[str, Any]:
        if isinstance(index, tuple):
            index = index[0]
        sample = self.get_data_info(index)
        image = self.samples[index]["image"]
        if self.size_transform is not None:
            image = self.size_transform(image, self.total_compression_ratio)
        sample["image"] = image
        return sample


class GEditDataProvider(ImageEditCoreEvalDataProvider):
    def __init__(self, cfg: GEditDataProviderConfig):
        super().__init__(cfg)
        self.cfg: GEditDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        if self.cfg.resolution in ["512F32MS", "512F64MS"]:
            resolution = 512
        elif self.cfg.resolution == "1024":
            resolution = 1024
        elif self.cfg.resolution == "2048":
            resolution = 2048
        else:
            raise ValueError(f"Resolution {self.cfg.resolution} is not supported")

        if self.cfg.resolution == "512F32MS":
            total_compression_ratio = 32
        elif self.cfg.resolution in ["512F64MS", "1024", "2048"]:
            total_compression_ratio = 64
        else:
            raise ValueError(f"Resolution {self.cfg.resolution} is not supported")

        dataset = GEditDataset(
            image_dir=self.cfg.data_dir,
            json_path=self.cfg.meta_path,
            num_samples=self.cfg.num_samples,
            seed=self.cfg.seed,
            total_compression_ratio=total_compression_ratio,
            size_transform=self.cfg.size_transform,
            resolution=resolution,
            language=self.cfg.language,
            task_type=self.cfg.task_type,
        )
        return dataset
