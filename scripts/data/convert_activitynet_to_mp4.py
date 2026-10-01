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
import shutil
from dataclasses import dataclass
from typing import Optional

import ray
from tqdm import tqdm

from dc_ai.apps.utils.config import get_config
from dc_ai.apps.utils.video import convert_to_mp4_h264


@dataclass
class ConvertConfig:
    num_processes: int = 16
    data_dir: str = "assets/dataset/activitynet/v1-2/test"


@ray.remote(num_cpus=1)
class Distributer:
    def __init__(self, cfg: ConvertConfig):
        self.cfg = cfg

        self.data_paths = []

        file_name_set = set(os.listdir(cfg.data_dir))
        for file_name in file_name_set:
            if file_name.endswith(".mp4"):
                continue
            elif file_name.endswith(".mkv") or file_name.endswith(".webm"):
                if os.path.splitext(file_name)[0] + ".mp4" not in file_name_set:
                    self.data_paths.append(os.path.join(cfg.data_dir, file_name))
            else:
                raise ValueError(f"Unsupported extention in {file_name}")

        self.idx = 0
        self.tqdm = tqdm(total=len(self.data_paths), desc="converting", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

    def get_next_task(self, process_id: int) -> Optional[str]:
        if self.idx == len(self.data_paths):
            return None
        self.tqdm.update()
        data_path = self.data_paths[self.idx]
        self.idx += 1
        return data_path


@ray.remote(num_cpus=1)
class Converter:
    def __init__(self, cfg: ConvertConfig, distributer: Distributer, process_id: int):
        self.cfg = cfg
        self.distributer = distributer
        self.process_id = process_id

    def convert(self, data_path: str) -> None:
        relative_path = os.path.relpath(data_path, self.cfg.data_dir)
        output_path = os.path.join(self.cfg.data_dir, os.path.splitext(relative_path)[0] + ".mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        convert_to_mp4_h264(data_path, output_path)

    def work(self):
        while True:
            data_path = ray.get(self.distributer.get_next_task.remote(self.process_id))
            if data_path is None:
                break
            self.convert(data_path)


def main():
    cfg = get_config(ConvertConfig)
    cfg.data_dir = os.path.expanduser(cfg.data_dir)

    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg is required on PATH to convert ActivityNet videos")

    ray.init()
    distributer = Distributer.remote(cfg)
    converters = [Converter.remote(cfg, distributer, process_id) for process_id in range(cfg.num_processes)]
    ray.get([converter.work.remote() for converter in converters])


if __name__ == "__main__":
    main()

"""
RAY_DEDUP_LOGS=0 python -m scripts.data.convert_activitynet_to_mp4
"""
