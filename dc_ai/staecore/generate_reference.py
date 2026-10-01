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

import torch
from omegaconf import MISSING
from tqdm import tqdm

from ..apps.metrics.fvd.fvd import FVDStats, FVDStatsConfig
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_init, get_dist_local_rank, is_master
from .data_provider.activitynet import ActivityNet13TestDataProviderConfig, ActivityNetDataProvider


@dataclass
class GenerateReferenceConfig:
    fvd: FVDStatsConfig = field(default_factory=FVDStatsConfig)

    # dataset
    dataset: str = MISSING
    activitynet_1_3_test: ActivityNet13TestDataProviderConfig = field(
        default_factory=ActivityNet13TestDataProviderConfig
    )


def main():
    cfg = get_config(GenerateReferenceConfig)

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    if cfg.dataset == "activitynet_1_3_test":
        data_provider = ActivityNetDataProvider(cfg.activitynet_1_3_test)
    else:
        raise ValueError(f"dataset {cfg.dataset} is not supported")

    fvd_stats = FVDStats(cfg.fvd)

    def add_data_from_dataloader(data_loader):
        for batch in tqdm(data_loader, disable=not is_master()):
            samples = batch["video"] if isinstance(batch, dict) else batch[0]
            samples = samples.cuda()
            fvd_stats.add_data(samples)

    add_data_from_dataloader(data_provider.data_loader)
    fvd_stats.get_stats()


if __name__ == "__main__":
    main()
