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

import os
from dataclasses import dataclass

from torch.utils.data import Dataset

from ...apps.utils.image import MultiResolutionImageFolder
from .base import AECoreDataProvider, AECoreDataProviderConfig


@dataclass
class MJHQDataProviderConfig(AECoreDataProviderConfig):
    name: str = "MJHQ"


@dataclass
class MJHQEvalDataProviderConfig(MJHQDataProviderConfig):
    name: str = "MJHQEval"
    resolution: int = 256
    data_dir: str = "~/dataset/MJHQ-30K/imgs"
    vlm_caption_path: str = "assets/data/vlm_caption/OpenGVLab_InternVL2-26B_MJHQEval.csv"

    def __post_init__(self):
        self.fid_ref_path: str = f"assets/data/fid/mjhq_{self.resolution}.npz"


class MJHQDataProvider(AECoreDataProvider):
    def __init__(self, cfg: MJHQDataProviderConfig):
        super().__init__(cfg)
        self.cfg: MJHQDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        size_transform, transform = self.build_transform()
        dataset = MultiResolutionImageFolder(
            os.path.join(self.cfg.data_dir), size_transform, transform, vlm_caption=self.vlm_caption
        )
        return dataset
