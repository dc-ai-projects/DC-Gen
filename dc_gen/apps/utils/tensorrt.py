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
import re
from typing import Any


def load_tensorrt_result(result_path: str) -> dict[str, Any]:
    result = {}
    with open(result_path, "r") as f:
        for line in f.readlines():
            if "Throughput" in line:
                re_result = re.findall(r"Throughput: (\d+.\d+) qps", line)
                result["throughput"] = float(re_result[0])
            if "[I] GPU Compute Time" in line:
                re_result = re.findall(
                    r"\[I\] GPU Compute Time: min = (\d+.\d+) ms, max = (\d+.\d+) ms, mean = (\d+.\d+) ms, median = (\d+.\d+) ms",
                    line,
                )
                result["min_latency"], result["max_latency"], result["mean_latency"], result["median_latency"] = map(
                    float, re_result[0]
                )
    return result


def get_tensorrt_result(onnx_export_path: str, result_path: str) -> dict[str, Any]:
    result_loaded = False
    if os.path.exists(result_path):
        result = load_tensorrt_result(result_path)
        if len(result) > 0:
            result_loaded = True

    if not result_loaded:
        cmd = f"trtexec --iterations=1000 --duration=0 --fp16 --device=0 --onnx={onnx_export_path} > {result_path}"
        print(cmd)
        os.popen(cmd).read()
        result = load_tensorrt_result(result_path)

    return result
