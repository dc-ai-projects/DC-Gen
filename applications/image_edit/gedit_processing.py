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

from tqdm import tqdm

from dc_ai.apps.utils.config import get_config


@dataclass
class EditBenchDataProcessingConfig:
    bench_name: str = "GEdit"
    input_dir: str = "assets/data/GEdit_raw"
    output_dir: str = "assets/data/GEdit"


def process_gedit(cfg: EditBenchDataProcessingConfig):
    from datasets import load_dataset

    dataset = load_dataset("arrow", data_files=f"{cfg.input_dir}/data-*-of-00009.arrow")
    dataset = dataset["train"]

    assert len(dataset) == 1212, "Dataset Size Error!"

    os.makedirs(os.path.join(cfg.output_dir, "images"), exist_ok=True)
    meta = {}

    for data in tqdm(dataset):
        img = data["input_image"]

        idx = 0
        while f"{data['key']}_{str(idx)}" in meta:
            idx += 1
        meta[f"{data['key']}_{str(idx)}"] = {
            "prompts": [data["instruction"]],
            "language": data["instruction_language"],
            "task_type": data["task_type"],
            "height": img.height,
            "width": img.width,
        }

        img.save(os.path.join(cfg.output_dir, "images", f"{data['key']}_{str(idx)}.png"))

    with open(os.path.join(cfg.output_dir, "meta_data.json"), "w") as json_file:
        json.dump(meta, json_file, indent=4)


def main():
    cfg = get_config(EditBenchDataProcessingConfig)
    if cfg.bench_name == "GEdit":
        process_gedit(cfg)
    else:
        raise ValueError(f"Benchmark {cfg.bench_name} is not supported.")


if __name__ == "__main__":
    main()
