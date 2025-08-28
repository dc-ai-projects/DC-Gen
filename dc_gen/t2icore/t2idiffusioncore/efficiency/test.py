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
from typing import Any

import torch
from omegaconf import OmegaConf

from ....apps.utils.config import get_config
from ....apps.utils.dist import dist_init, get_dist_local_rank
from ....apps.utils.tensorrt import get_tensorrt_result
from ....models.utils.network import get_params_num
from ..models.sana_t2i import SanaT2I
from .model import ModelConfig
from .onnx_export import onnx_export_diffusion


def test_num_params(model_cfg: ModelConfig):
    if model_cfg.model == "sana_t2i":
        model = SanaT2I(model_cfg.sana_t2i)
    else:
        raise ValueError(f"model {model_cfg.model} is not supported")
    return get_params_num(model)


def test_onnx_trt(cfg: ModelConfig) -> dict[str, Any]:
    onnx_export_path = onnx_export_diffusion(cfg)
    torch.cuda.empty_cache()
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    result_path = f"{onnx_export_path[:-5]}_{device_name}.txt"
    result = get_tensorrt_result(onnx_export_path, result_path)
    result["throughput"] *= cfg.batch_size
    return result


@dataclass
class TestDiffusionEfficiencyConfig:
    test_num_params: bool = False
    test_inference_throughput: bool = False
    test_inference_latency: bool = False
    test_nfe: bool = False


def main():
    dist_init()
    torch.cuda.set_device(get_dist_local_rank())
    cfg = get_config(TestDiffusionEfficiencyConfig)

    base_model_cfg = OmegaConf.structured(ModelConfig)

    model_backbone_cfgs = {
        backbone_name: OmegaConf.merge(base_model_cfg, backbone_cfg)
        for backbone_name, backbone_cfg in {
            "sana_t2i_1.5_1.6b": OmegaConf.create(
                {
                    "model": "sana_t2i",
                    "sana_t2i": {
                        "depth": 20,
                        "hidden_size": 2240,
                        "num_heads": 20,
                        "qk_norm": True,
                        "cross_norm": True,
                    },
                    "large": True,
                }
            ),
            "sana_t2i_1.5_4.8b": OmegaConf.create(
                {
                    "model": "sana_t2i",
                    "sana_t2i": {
                        "depth": 60,
                        "hidden_size": 2240,
                        "num_heads": 20,
                        "qk_norm": True,
                        "cross_norm": True,
                    },
                    "large": True,
                }
            ),
        }.items()
    }

    text_input_cfgs = {
        "sana_t2i_1.5_1.6b": OmegaConf.create({"0": {"name": "google/gemma-2-2b-it"}}),
        "sana_t2i_1.5_4.8b": OmegaConf.create({"0": {"name": "google/gemma-2-2b-it"}}),
    }

    model_input_cfgs = {
        "1024_dc_f32_p1": OmegaConf.create({"patch_size": 1, "in_channels": 32, "input_size": 32}),
        "1024_dc_f64_p1": OmegaConf.create({"patch_size": 1, "in_channels": 128, "input_size": 16}),
    }

    model_cfgs = {
        f"{input_name}@{backbone_name}": OmegaConf.merge(
            model_backbone_cfgs[backbone_name],
            {"text_encoders": text_input_cfgs[backbone_name]},
            {model_backbone_cfgs[backbone_name].model: model_input_cfgs[input_name]},
        )
        for input_name, backbone_name in [
            ("1024_dc_f32_p1", "sana_t2i_1.5_1.6b"),
            ("1024_dc_f64_p1", "sana_t2i_1.5_1.6b"),
            ("1024_dc_f32_p1", "sana_t2i_1.5_4.8b"),
            ("1024_dc_f64_p1", "sana_t2i_1.5_4.8b"),
        ]
    }

    if cfg.test_num_params:
        result_path = os.path.join(os.path.dirname(__file__), "num_params.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
        else:
            result = {}
        for model_name, model_cfg in model_cfgs.items():
            if model_name not in result:
                result[model_name] = test_num_params(OmegaConf.to_object(model_cfg))
            print(f"{model_name} num params: {result[model_name]}")
        with open(result_path, "w") as f:
            json.dump(result, f)

    if cfg.test_inference_latency:
        for model_name, model_cfg in model_cfgs.items():
            onnx_trt_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(ModelConfig),
                    OmegaConf.to_container(model_cfg),
                    {"batch_size": 2},
                )
            )
            print(f"{model_name} inference latency: {test_onnx_trt(onnx_trt_cfg)}")

    if cfg.test_inference_throughput:
        num_tokens_to_batch_size = {
            "sana_t2i_1.5_1.6b": {16: 256, 32: 64},
            "sana_t2i_1.5_4.8b": {16: 128, 32: 32},
        }
        for model_name, model_cfg in model_cfgs.items():
            input_name, backbone_name = model_name.split("@")
            model = model_cfg.model
            onnx_trt_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(ModelConfig),
                    OmegaConf.to_container(model_cfg),
                    {
                        "batch_size": num_tokens_to_batch_size[backbone_name][
                            model_cfg[model]["input_size"] // model_cfg[model]["patch_size"]
                        ],
                    },
                )
            )
            print(f"{model_name} inference throughput: {test_onnx_trt(onnx_trt_cfg)}")

    if cfg.test_nfe:
        result_path = os.path.join(os.path.dirname(__file__), "num_functional_evaluations.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
        else:
            result = {}
        for model_name, model_cfg in model_cfgs.items():
            if model_name not in result:
                result[model_name] = test_nfe(OmegaConf.to_object(model_cfg))
            print(f"{model_name} nfe: {result[model_name]}")
        with open(result_path, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    main()
