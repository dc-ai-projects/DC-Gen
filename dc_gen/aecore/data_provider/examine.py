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
import sys
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas
import ray
from omegaconf import MISSING, OmegaConf
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

from ...apps.utils.image import IdentityTransform
from .collection import possible_datasets


@dataclass
class DataProviderExaminationConfig:
    dataset: str = MISSING
    use_ray: bool = True
    num_processes: int = 16
    num_samples_per_task: Optional[int] = None
    task_id: int = 0
    chunk_size: int = 10000


class DataProviderExaminationDistributer:
    def __init__(self, cfg: DataProviderExaminationConfig):
        self.cfg = cfg

        possible_datasets = self.get_possible_datasets()
        if cfg.dataset in possible_datasets:
            self.dataset = possible_datasets[cfg.dataset](
                size_transform=IdentityTransform(), transform=IdentityTransform()
            )
        else:
            raise ValueError(f"dataset {cfg.dataset} is not supported")

        if cfg.num_samples_per_task is not None:
            num_tasks = (len(self.dataset) - 1) // cfg.num_samples_per_task + 1
            print(f"num_tasks {num_tasks}")
            self.start = min(len(self.dataset), cfg.task_id * cfg.num_samples_per_task)
            self.end = min(len(self.dataset), self.start + cfg.num_samples_per_task)
        else:
            self.start, self.end = 0, len(self.dataset)

        self.new_chunk_start = self.start
        self.process_task_id_list = -np.ones(cfg.num_processes, dtype=int)
        self.process_task_end_list = -np.ones(cfg.num_processes, dtype=int)

        self.tqdm = tqdm(
            total=self.end - self.start,
            desc="examining",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            file=sys.stdout,
        )

    def get_possible_datasets(self) -> dict[str, type[Dataset]]:
        return possible_datasets

    def get_dataset(self):
        return self.dataset

    def get_next_task(self, process_id: int) -> Optional[int]:
        if self.process_task_id_list[process_id] == self.process_task_end_list[process_id]:
            if self.new_chunk_start >= self.end:
                return None
            self.process_task_id_list[process_id] = self.new_chunk_start
            self.new_chunk_start = min(self.end, self.new_chunk_start + self.cfg.chunk_size)
            self.process_task_end_list[process_id] = self.new_chunk_start
        self.tqdm.update()
        task_id = self.process_task_id_list[process_id]
        self.process_task_id_list[process_id] += 1
        return task_id.item()


@ray.remote(num_cpus=1)
class DataProviderExaminationDistributerRemote(DataProviderExaminationDistributer):
    pass


class DataProviderExaminator:
    def __init__(
        self, cfg: DataProviderExaminationConfig, distributer: DataProviderExaminationDistributerRemote, process_id: int
    ):
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.cfg = cfg
        self.distributer = distributer
        self.process_id = process_id
        self.dataset = ray.get(self.distributer.get_dataset.remote())

    def examine(self, idx: int):
        try:
            sample: tuple[Image.Image, Any] = self.dataset[idx]
            image = sample[0]
            return {"H": image.size[1], "W": image.size[0], "mode": image.mode}
        except:
            return {"H": 0, "W": 0, "mode": "corrupted"}

    def work(self):
        results = {}
        while True:
            idx = ray.get(self.distributer.get_next_task.remote(self.process_id))
            if idx is None:
                break
            results[idx] = self.examine(idx)
        data_frame = pandas.DataFrame.from_dict(results, orient="index")
        return data_frame


@ray.remote(num_cpus=1)
class DataProviderExaminatorRemote(DataProviderExaminator):
    pass


def main():
    cfg: DataProviderExaminationConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DataProviderExaminationConfig), OmegaConf.from_cli())
    )

    ray.init(ignore_reinit_error=True, num_cpus=cfg.num_processes + 1)
    distributer = DataProviderExaminationDistributerRemote.remote(cfg)

    os.makedirs("assets/data/examination", exist_ok=True)
    if cfg.num_samples_per_task is not None:
        result_path = os.path.join(
            "assets/data/examination", f"{cfg.dataset}_{cfg.num_samples_per_task}_{cfg.task_id}.csv"
        )
    else:
        result_path = os.path.join("assets/data/examination", f"{cfg.dataset}.csv")

    if cfg.use_ray:
        examinators = [
            DataProviderExaminatorRemote.remote(cfg, distributer, process_id) for process_id in range(cfg.num_processes)
        ]
        data_frames = ray.get([examinator.work.remote() for examinator in examinators])
        data_frame = pandas.concat(data_frames).sort_index()
    else:
        examinator = DataProviderExaminator(cfg, distributer, 0)
        data_frame = examinator.work()
    data_frame.to_csv(result_path)


if __name__ == "__main__":
    main()

"""
RAY_DEDUP_LOGS=0 python -m dc_gen.aecore.data_provider.examine dataset=ImageNet_train
RAY_DEDUP_LOGS=0 python -m dc_gen.aecore.data_provider.examine dataset=ImageNet_eval
"""
