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
from dataclasses import dataclass

import numpy as np
from omegaconf import MISSING

from ...apps.utils.config import get_config


@dataclass
class ReportGEditScoreConfig:
    raw_results_path: str = MISSING


def main():
    cfg: ReportGEditScoreConfig = get_config(ReportGEditScoreConfig)

    with open(cfg.raw_results_path, "rb") as json_file:
        full_info = json.load(json_file)

    results = {
        "G_SC_en": [],
        "G_PQ_en": [],
        "G_O_en": [],
        "G_SC_cn": [],
        "G_PQ_cn": [],
        "G_O_cn": [],
    }

    for key, value in full_info.items():
        if type(value) == dict:
            results[f"{key}_en"], results[f"{key}_cn"] = [], []
            for item in value["detailed_info"]:
                language = item["language"]
                sementics_score, quality_score, overall_score = (
                    item["sementics_score"],
                    item["quality_score"],
                    item["overall_score"],
                )
                results[f"G_SC_{language}"].append(sementics_score)
                results[f"G_PQ_{language}"].append(quality_score)
                results[f"G_O_{language}"].append(overall_score)
                results[f"{key}_{language}"].append(overall_score)

    for key, value in results.items():
        print(f"{key}: {np.mean(value)}")


if __name__ == "__main__":
    main()

"""
python -m dc_ai.imageeditcore.metrics_offline.report_gedit_score \
    raw_results_path=tmp_qwen_image_edit/qwen25vl/result_info_all.json
python -m dc_ai.imageeditcore.metrics_offline.report_gedit_score \
    raw_results_path=tmp_dc_qwen_image/qwen25vl/result_info_all.json
"""
