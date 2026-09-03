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

import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ...apps.data_provider.sampler import DistributedRangedSampler
from ...apps.metrics.clip_score import CLIPScoreStats
from ...apps.utils.dist import dist_init, get_dist_local_rank, get_dist_rank, get_dist_size, is_master
from ...apps.utils.image import load_image


class ComputeCLIPScoreDataset(Dataset):
    def __init__(self, setting: str, data_dir: str, suffix: str = ".jpg"):
        super().__init__()
        self.setting = setting
        self.data_dir = data_dir
        self.suffix = suffix
        if setting == "MJHQ-30K":
            with open(os.path.expanduser("~/dataset/MJHQ-30K/meta_data.json"), "r") as f:
                self.meta_data = list(json.load(f).items())
        else:
            raise ValueError(f"Setting {setting} not supported")

    def __len__(self):
        if self.setting == "MJHQ-30K":
            return len(self.meta_data)
        else:
            raise ValueError(f"Setting {self.setting} not supported")

    def __getitem__(self, index: int):
        if self.setting == "MJHQ-30K":
            image_path = os.path.join(self.data_dir, f"{self.meta_data[index][0]}{self.suffix}")
            image = load_image(image_path)
            image = np.array(image)
            image = torch.from_numpy(image).permute(2, 0, 1)
            return image, self.meta_data[index][1]["prompt"]
        else:
            raise ValueError(f"Setting {self.setting} not supported")


@dataclass
class ComputeCLIPScoreConfig:
    setting: str = "MJHQ-30K"
    data_dir: str = MISSING
    suffix: str = ".jpg"
    batch_size: int = 100
    num_workers: int = 8


def main():
    cfg: ComputeCLIPScoreConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ComputeCLIPScoreConfig), OmegaConf.from_cli())
    )

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())
    device = torch.device("cuda")

    dataset = ComputeCLIPScoreDataset(setting=cfg.setting, data_dir=cfg.data_dir, suffix=cfg.suffix)

    sampler = DistributedRangedSampler(dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
        sampler=sampler,
    )

    if cfg.setting == "MJHQ-30K":
        clip_score_stats = CLIPScoreStats()
    else:
        raise ValueError(f"Setting {cfg.setting} not supported")

    for image, prompt in tqdm(data_loader):
        clip_score_stats.update(image.to(device), prompt)

    if cfg.setting == "MJHQ-30K":
        clip_score = clip_score_stats.compute()
    else:
        raise ValueError(f"Setting {cfg.setting} not supported")

    if is_master():
        print(f"clip_score: {clip_score}")


if __name__ == "__main__":
    main()
