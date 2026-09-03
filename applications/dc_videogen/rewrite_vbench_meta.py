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

from dc_ai.apps.utils.config import get_config


@dataclass
class RewriteVBenchMetaConfig:
    raw_path: str = "assets/data/vbench/VBench_full_info.json"
    extended_path: str = "assets/data/vbench/vbench_extended.txt"
    save_path: str = "assets/data/vbench/VBench_extended_full_info.json"
    task_type: str = "t2v"


def rewrite_meta_t2v(raw_path, extended_path, save_path):
    with open(raw_path, "r") as json_file:
        meta = json.load(json_file)

    meta_extended = {}
    dimensions_list = []

    for item in meta:
        dimension = item["dimension"][0]
        if dimension not in dimensions_list:
            dimensions_list.append(dimension)

    with open(extended_path, "r") as txt_file:
        lines = txt_file.readlines()

    for vid, item in enumerate(meta):
        dimension = item["dimension"][0]

        dimension_id = dimensions_list.index(dimension)
        if str(dimension_id) not in meta_extended:
            meta_extended[str(dimension_id)] = {"dimension": dimension, "videos": []}

        new_item = {
            "video_paths": [],
            "prompt": item["prompt_en"],
            "extended_prompt": lines[vid].strip(),
            "dimension": item["dimension"],
        }
        if "auxiliary_info" in item:
            new_item["auxiliary_info"] = item["auxiliary_info"]

        for i in range(5):
            new_item["video_paths"].append(f"video_{vid}_{i}.mp4")

        meta_extended[str(dimension_id)]["videos"].append(new_item)

    with open(save_path, "w") as json_file:
        json.dump(meta_extended, json_file, indent=4)


def rewrite_meta_i2v(raw_path, save_path):
    with open(raw_path, "r") as json_file:
        meta = json.load(json_file)

    meta_extended = {
        "0": {
            "dimension": "i2v_subject",
            "videos": [],
        },
        "1": {
            "dimension": "i2v_background",
            "videos": [],
        },
        "2": {
            "dimension": "camera_motion",
            "videos": [],
        },
    }

    for vid, item in enumerate(meta):
        if item["dimension"][0] == "i2v_subject":
            category_id = 0
        elif item["dimension"][0] == "i2v_background":
            category_id = 1
        elif item["dimension"][0] == "camera_motion":
            category_id = 2
        else:
            raise ValueError(f"dimension {item['dimension'][0]} not supported")

        meta_extended[str(category_id)]["videos"].append(
            {
                "prompt": item["prompt_en"],
                "extended_prompt": item["extended_prompt"],
                "image_name": item["image_name"],
                "dimension": item["dimension"],
                "video_paths": [],
            }
        )

        dimension = item["dimension"][0]
        prompt = item["prompt_en"]
        for i in range(5):
            meta_extended[str(category_id)]["videos"][-1]["video_paths"].append(
                f"video_{dimension}_{vid}_{i}_{prompt}.mp4"
            )

    with open(save_path, "w") as json_file:
        json.dump(meta_extended, json_file, indent=4)


def main():
    cfg = get_config(RewriteVBenchMetaConfig)
    if cfg.task_type == "t2v":
        rewrite_meta_t2v(cfg.raw_path, cfg.extended_path, cfg.save_path)
    elif cfg.task_type == "i2v":
        rewrite_meta_i2v(cfg.raw_path, cfg.save_path)
    else:
        raise NotImplementedError(f"{cfg.task_type} is not supported by vbench")


if __name__ == "__main__":
    main()
