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
from typing import Any

import pandas
import ray
from omegaconf import MISSING
from tqdm import tqdm

from ...apps.utils.config import get_config
from ...apps.utils.multiprocess import Distributor, MultiProcessConfig, SubtaskStatus, Worker
from ...apps.utils.video import VideoLoader
from .collection import possible_datasets


@dataclass
class VideoDataProviderExaminationConfig(MultiProcessConfig):
    dataset: str = MISSING
    num_cpus_per_worker: int = 10
    num_samples_per_subtask: int = 1000
    merge_results: bool = False


class VideoDataProviderExaminationDistributer(Distributor):
    def __init__(self, cfg: VideoDataProviderExaminationConfig):
        super().__init__(cfg)
        self.cfg: VideoDataProviderExaminationConfig

    def build_subtasks(self) -> dict[int, Any]:
        dataset = possible_datasets[self.cfg.dataset]()
        num_samples = len(dataset)
        print(f"{num_samples=}")
        num_subtasks = (num_samples - 1) // self.cfg.num_samples_per_subtask + 1
        if self.cfg.num_subtasks_per_task is not None and self.cfg.task_id is not None:
            start_id = min(self.cfg.task_id * self.cfg.num_subtasks_per_task, num_subtasks)
            end_id = min(start_id + self.cfg.num_subtasks_per_task, num_subtasks)
        else:
            start_id, end_id = 0, num_subtasks
        subtasks = {subtask_id: subtask_id for subtask_id in list(range(start_id, end_id))}
        return subtasks


class VideoDataProviderExaminator(Worker):
    def __init__(
        self,
        cfg: VideoDataProviderExaminationConfig,
        distributer: VideoDataProviderExaminationDistributer,
        worker_id: int,
    ):
        super().__init__(cfg, distributer, worker_id)
        self.cfg: DataProviderExaminationConfig

        self.dataset = possible_datasets[cfg.dataset]()
        if cfg.dataset == "UCF101":
            with open(os.path.expanduser("~/dataset/ucf101/testlist01.txt"), "r") as f:
                eval_split_list = []
                for line in f.readlines():
                    eval_split_list.append(line.rstrip()[:-4] + ".mp4")
                self.eval_split_set = set(eval_split_list)

    def do_subtask(self, subtask: int) -> SubtaskStatus:
        # if self.cfg.dataset == "FusionX480P":
        #     return {"T": 81, "H": 480, "W": 832, "fps": 16.0, "duration": 5.0625}
        start_id = min(self.cfg.num_samples_per_subtask * subtask, len(self.dataset))
        end_id = min(start_id + self.cfg.num_samples_per_subtask, len(self.dataset))
        results = {}
        for i in tqdm(
            range(start_id, end_id),
            desc=f"doing subtask {subtask}",
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            file=sys.stdout,
        ):
            sample: dict[str, Any] = self.dataset[i]
            video: VideoLoader = sample["video"]
            try:
                frames = video.get_all_frames()
                frame_count = len(frames)
                if frame_count > 0:
                    H, W = frames[0].size[1], frames[0].size[0]
                else:
                    H, W = 0, 0
                fps = video.get_fps()
            except:
                frame_count, fps, H, W = 0, 0, 0, 0
            metadata = {"T": frame_count, "H": H, "W": W, "fps": fps, "duration": frame_count / fps if fps != 0 else 0}
            if self.cfg.dataset == "UCF101":
                metadata["eval"] = os.path.relpath(video.path, self.dataset.root) in self.eval_split_set
            results[i] = metadata
        data_frame = pandas.DataFrame.from_dict(results, orient="index")
        data_frame.to_csv(os.path.join(self.cfg.run_dir, f"{subtask:06d}.csv"))
        return SubtaskStatus.COMPLETED


def main():
    cfg = get_config(VideoDataProviderExaminationConfig)
    if cfg.merge_results:
        result_path = os.path.join("assets/data/examination", f"{cfg.dataset}.csv")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        data_frame_list: list[pandas.DataFrame] = []
        for file_name in os.listdir(cfg.run_dir):
            if not file_name.endswith(".csv"):
                continue
            file_path = os.path.join(cfg.run_dir, file_name)
            data_frame_list.append(pandas.read_csv(file_path, index_col=0))
        data_frame = pandas.concat(data_frame_list).sort_index()
        data_frame.to_csv(result_path)
        return

    if cfg.use_ray:
        ray.init(ignore_reinit_error=True)

        @ray.remote(num_cpus=cfg.num_cpus_per_worker)
        class VideoDataProviderExaminatorRemote(VideoDataProviderExaminator):
            pass

        @ray.remote(num_cpus=1)
        class VideoDataProviderExaminationDistributerRemote(VideoDataProviderExaminationDistributer):
            pass

        distributer = VideoDataProviderExaminationDistributerRemote.remote(cfg)
        workers = [
            VideoDataProviderExaminatorRemote.remote(cfg, distributer, worker_id)
            for worker_id in range(cfg.num_workers)
        ]
        ray.get([worker.work.remote() for worker in workers])
    else:
        distributor = VideoDataProviderExaminationDistributer(cfg)
        worker = VideoDataProviderExaminator(cfg, distributor, 0)
        worker.work()


if __name__ == "__main__":
    main()
