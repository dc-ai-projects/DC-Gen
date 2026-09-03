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
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import MISSING

from ..utils.tar import generate_tar
from ..utils.zip import generate_zip


@dataclass
class ArchivesGeneratorConfig:
    save_dir: str = MISSING
    task_id: int = 0
    num_archives_per_task: int = MISSING
    archive_format: str = "tar"
    seed: int = 0


class ArchivesGenerator:
    def __init__(self, cfg: ArchivesGeneratorConfig):
        self.cfg = cfg
        self.setup_env()
        self.setup_seed()
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def setup_env(self) -> None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def setup_seed(self) -> None:
        seed = self.cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_save_path(self, index: int) -> str:
        raise NotImplementedError

    def generate_data_for_single_archive(self, index: int) -> tuple[dict[str, str], list[tuple[str, dict[str, Any]]]]:
        """
        data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
        data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
        """
        raise NotImplementedError

    def generate_single_archive(self, index: int):
        save_path = self.get_save_path(index)
        if os.path.exists(save_path):
            print(f"Skipping {save_path} because it already exists")
            return
        data_format, data_list = self.generate_data_for_single_archive(index)
        if self.cfg.archive_format == "tar":
            generate_tar(data_format, data_list, save_path)
        elif self.cfg.archive_format == "zip":
            generate_zip(data_format, data_list, save_path)
        else:
            raise ValueError(f"Unsupported archive format: {self.cfg.archive_format}")

    def generate(self):
        for index in range(
            self.cfg.task_id * self.cfg.num_archives_per_task, (self.cfg.task_id + 1) * self.cfg.num_archives_per_task
        ):
            self.generate_single_archive(index)
