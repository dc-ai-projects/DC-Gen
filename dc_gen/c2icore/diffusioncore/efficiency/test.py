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
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from omegaconf import MISSING, OmegaConf
from timm.layers.config import set_fused_attn
from torch.nn import functional as F
from torchprofile import profile_macs

from ....apps.utils.dist import dist_init, get_dist_local_rank, is_dist_initialized, is_master
from ....apps.utils.tensorrt import get_tensorrt_result
from ....models.utils.network import get_params_num
from ..models.base import BaseDiffusionModel
from ..models.dit import DiT, DiTConfig
from ..models.sana_cls import SanaCls, SanaClsConfig
from ..models.uvit import UViT, UViTConfig
from .onnx_export import OnnxExportDiffusionConfig, onnx_export_diffusion
from .training import TestDiffusionTrainingConfig, test_diffusion_training


@dataclass
class ModelConfig:
    model: str = MISSING
    dit: DiTConfig = field(default_factory=DiTConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)
    sana_cls: SanaClsConfig = field(default_factory=SanaClsConfig)
    large: bool = False
    fsdp: bool = False


def test_num_params(model_cfg: ModelConfig):
    if model_cfg.model == "dit":
        model = DiT(model_cfg.dit)
    elif model_cfg.model == "uvit":
        model = UViT(model_cfg.uvit)
    elif model_cfg.model == "sana_cls":
        model = SanaCls(model_cfg.sana_cls)
    else:
        raise ValueError(f"model {model_cfg.model} is not supported")
    return get_params_num(model)


class OnnxDiffusionModel(torch.nn.Module):
    def __init__(self, model: BaseDiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, x, t, y):
        return self.model.forward_without_cfg(x, t, y)[0]


def naive_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    if attn_mask is not None or dropout_p != 0.0 or is_causal:
        raise NotImplementedError
    scale = scale or query.shape[-1] ** -0.5
    attn = torch.matmul(query, key.transpose(-1, -2)) * scale
    attn = F.softmax(attn, -1, _stacklevel=5)
    out = torch.matmul(attn, value)
    return out


def test_macs(model_cfg: ModelConfig) -> int:
    device = torch.device("cpu")
    if model_cfg.model == "dit":
        model = DiT(model_cfg.dit).to(device)
        x = torch.randn(1, model.cfg.in_channels, model.cfg.input_size, model.cfg.input_size, device=device)
        t = torch.rand(1, device=device)
        y = torch.randint(0, max(model.cfg.num_classes, 1), (1,), device=device)
        return profile_macs(OnnxDiffusionModel(model), (x, t, y))
    elif model_cfg.model == "uvit":
        model = UViT(model_cfg.uvit).to(device)
        x = torch.randn(1, model.cfg.in_channels, model.cfg.input_size, model.cfg.input_size, device=device)
        t = torch.rand(1, device=device)
        y = torch.randint(0, max(model.cfg.num_classes, 1), (1,), device=device)
        return profile_macs(OnnxDiffusionModel(model), (x, t, y))
    elif model_cfg.model == "sana_cls":
        model = SanaCls(model_cfg.sana_cls).to(device)
        x = torch.randn(1, model.cfg.in_channels, model.cfg.input_size, model.cfg.input_size, device=device)
        t = torch.rand(1, device=device)
        y = torch.randint(0, max(model.cfg.num_classes, 1), (1,), device=device)
        return profile_macs(OnnxDiffusionModel(model), (x, t, y))
    else:
        raise ValueError(f"model {model_cfg.model} is not supported")


def test_onnx_trt(cfg: OnnxExportDiffusionConfig) -> dict[str, Any]:
    onnx_export_path = onnx_export_diffusion(cfg)
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    result_path = f"{onnx_export_path[:-5]}_{device_name}.txt"
    result = get_tensorrt_result(onnx_export_path, result_path)
    result["throughput"] *= cfg.batch_size
    return result


def test_nfe(model_cfg: ModelConfig):
    device = torch.device("cuda")
    if model_cfg.model == "dit":
        model_cfg.dit.count_nfe = True
        model = DiT(model_cfg.dit).to(device)
    elif model_cfg.model == "uvit":
        model_cfg.uvit.count_nfe = True
        model = UViT(model_cfg.uvit).to(device)
    elif model_cfg.model == "sana_cls":
        model_cfg.sana_cls.count_nfe = True
        model = SanaCls(model_cfg.sana_cls).to(device)
    else:
        raise ValueError(f"model {model_cfg.model} is not supported")
    model.generate(torch.tensor([0, 1, 2, 3], device=device))
    return model.nfe


@dataclass
class TestDiffusionEfficiencyConfig:
    onnx_trt: OnnxExportDiffusionConfig = field(default_factory=OnnxExportDiffusionConfig)
    training: TestDiffusionTrainingConfig = field(default_factory=TestDiffusionTrainingConfig)

    test_num_params: bool = False
    test_macs: bool = False
    test_inference_throughput: bool = False
    test_inference_latency: bool = False
    test_training_memory: bool = False
    test_training_throughput: bool = False
    test_nfe: bool = False


def main():
    dist_init()
    torch.cuda.set_device(get_dist_local_rank())
    cfg: TestDiffusionEfficiencyConfig = OmegaConf.merge(
        OmegaConf.structured(TestDiffusionEfficiencyConfig), OmegaConf.from_cli()
    )

    base_model_cfg = OmegaConf.structured(ModelConfig)

    model_backbone_cfgs = {
        backbone_name: OmegaConf.merge(base_model_cfg, backbone_cfg)
        for backbone_name, backbone_cfg in {
            "uvit_s": OmegaConf.create({"model": "uvit", "uvit": {"depth": 12, "hidden_size": 512, "num_heads": 8}}),
            "uvit_s_deep": OmegaConf.create(
                {"model": "uvit", "uvit": {"depth": 16, "hidden_size": 512, "num_heads": 8}}
            ),
            "uvit_m": OmegaConf.create({"model": "uvit", "uvit": {"depth": 16, "hidden_size": 768, "num_heads": 12}}),
            "uvit_l": OmegaConf.create(
                {"model": "uvit", "uvit": {"depth": 20, "hidden_size": 1024, "num_heads": 16}, "large": True}
            ),
            "uvit_h": OmegaConf.create(
                {"model": "uvit", "uvit": {"depth": 28, "hidden_size": 1152, "num_heads": 16}, "large": True}
            ),
            "uvit_2b": OmegaConf.create(
                {"model": "uvit", "uvit": {"depth": 28, "hidden_size": 2048, "num_heads": 32}, "large": True}
            ),
            "dit_s": OmegaConf.create({"model": "dit", "dit": {"depth": 12, "hidden_size": 384, "num_heads": 6}}),
            "dit_xl": OmegaConf.create(
                {"model": "dit", "dit": {"depth": 28, "hidden_size": 1152, "num_heads": 16}, "large": True}
            ),
            "sit_xl": OmegaConf.create(
                {
                    "model": "dit",
                    "dit": {
                        "depth": 28,
                        "hidden_size": 1152,
                        "num_heads": 16,
                        "learn_sigma": False,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                    },
                    "large": True,
                }
            ),
            "usit_h": OmegaConf.create(
                {
                    "model": "uvit",
                    "uvit": {
                        "depth": 28,
                        "hidden_size": 1152,
                        "num_heads": 16,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                        "num_inference_steps": 250,
                    },
                    "large": True,
                }
            ),
            "usit_2b": OmegaConf.create(
                {
                    "model": "uvit",
                    "uvit": {
                        "depth": 28,
                        "hidden_size": 2048,
                        "num_heads": 32,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                        "num_inference_steps": 250,
                    },
                    "large": True,
                }
            ),
            "usit_3b": OmegaConf.create(
                {
                    "model": "uvit",
                    "uvit": {
                        "depth": 56,
                        "hidden_size": 2048,
                        "num_heads": 32,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                        "num_inference_steps": 250,
                    },
                    "large": True,
                    "fsdp": True,
                }
            ),
            "usit_5b": OmegaConf.create(
                {
                    "model": "uvit",
                    "uvit": {
                        "depth": 84,
                        "hidden_size": 2048,
                        "num_heads": 32,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                        "num_inference_steps": 250,
                    },
                    "large": True,
                    "fsdp": True,
                }
            ),
            "sana_cls_xl": OmegaConf.create(
                {
                    "model": "sana_cls",
                    "sana_cls": {
                        "depth": 28,
                        "hidden_size": 1152,
                        "attention_head_dim": 32,
                        "train_scheduler": "SiTSampler",
                        "eval_scheduler": "ODE_dopri5",
                        "num_inference_steps": 250,
                    },
                    "large": True,
                }
            ),
        }.items()
    }

    model_input_cfgs = {
        "256_sd_f8_p2": OmegaConf.create({"patch_size": 2, "in_channels": 4, "input_size": 32}),
        "512_flux": OmegaConf.create({"patch_size": 2, "in_channels": 16, "input_size": 64}),
        "512_sd_f8_p2": OmegaConf.create({"patch_size": 2, "in_channels": 4, "input_size": 64}),
        "512_sd_f16_p2": OmegaConf.create({"patch_size": 2, "in_channels": 16, "input_size": 32}),
        "512_sd_f32_p1": OmegaConf.create({"patch_size": 1, "in_channels": 64, "input_size": 16}),
        "512_dc_f8_p1": OmegaConf.create({"patch_size": 1, "in_channels": 2, "input_size": 64}),
        "512_dc_f16_p1": OmegaConf.create({"patch_size": 1, "in_channels": 8, "input_size": 32}),
        "512_dc_f32_p1": OmegaConf.create({"patch_size": 1, "in_channels": 32, "input_size": 16}),
        "512_dc_f64_p1": OmegaConf.create({"patch_size": 1, "in_channels": 128, "input_size": 8}),
        "512_dc_f128_p1": OmegaConf.create({"patch_size": 1, "in_channels": 512, "input_size": 4}),
        "1024_flux": OmegaConf.create({"patch_size": 2, "in_channels": 16, "input_size": 128}),
    }

    model_cfgs = {
        f"{input_name}@{backbone_name}": OmegaConf.merge(
            model_backbone_cfgs[backbone_name], {model_backbone_cfgs[backbone_name].model: model_input_cfgs[input_name]}
        )
        for input_name, backbone_name in [
            # ("256_sd_f8_p2", "uvit_h"),
            ("512_flux", "uvit_s"),
            ("512_sd_f8_p2", "uvit_s"),
            ("512_sd_f16_p2", "uvit_s"),
            ("512_sd_f32_p1", "uvit_s"),
            ("512_dc_f32_p1", "uvit_s"),
            ("512_dc_f64_p1", "uvit_s"),
            ("512_flux", "dit_xl"),
            ("512_sd_f8_p2", "dit_xl"),
            ("512_dc_f8_p1", "dit_xl"),
            ("512_dc_f16_p1", "dit_xl"),
            ("512_dc_f32_p1", "dit_xl"),
            ("512_dc_f64_p1", "dit_xl"),
            ("512_dc_f128_p1", "dit_xl"),
            ("512_flux", "uvit_h"),
            ("512_sd_f8_p2", "uvit_h"),
            ("512_dc_f32_p1", "uvit_h"),
            ("512_dc_f64_p1", "uvit_h"),
            ("512_flux", "uvit_2b"),
            ("512_sd_f8_p2", "uvit_2b"),
            ("512_dc_f32_p1", "uvit_2b"),
            ("512_dc_f64_p1", "uvit_2b"),
            ("512_dc_f32_p1", "sit_xl"),
            ("512_dc_f32_p1", "sana_cls_xl"),
            ("512_dc_f32_p1", "usit_h"),
            ("512_dc_f64_p1", "usit_h"),
            ("512_dc_f64_p1", "usit_2b"),
            ("512_dc_f64_p1", "usit_3b"),
            ("512_dc_f64_p1", "usit_5b"),
            ("1024_flux", "dit_s"),
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

    if cfg.test_macs:
        result_path = os.path.join(os.path.dirname(__file__), "macs.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                result = json.load(f)
        else:
            result = {}
        set_fused_attn(False)
        F.scaled_dot_product_attention = naive_scaled_dot_product_attention
        for model_name, model_cfg in model_cfgs.items():
            if model_name not in result:
                result[model_name] = test_macs(OmegaConf.to_object(model_cfg))
            print(f"{model_name} macs: {result[model_name]/10**9} G")
        with open(result_path, "w") as f:
            json.dump(result, f)

    if cfg.test_inference_latency:
        for model_name, model_cfg in model_cfgs.items():
            onnx_trt_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(cfg.onnx_trt),
                    OmegaConf.to_container(model_cfg),
                    {"batch_size": 2},
                )
            )
            print(f"{model_name} inference latency: {test_onnx_trt(onnx_trt_cfg)}")

    if cfg.test_inference_throughput:
        num_tokens_to_batch_size = {
            "uvit_s": {32: 512, 16: 2048, 8: 8192},
            "uvit_h": {32: 128, 16: 512, 8: 2048},
            "uvit_2b": {32: 128, 16: 512, 8: 2048},
            "usit_h": {32: 128, 16: 512, 8: 2048},
            "usit_2b": {32: 128, 16: 512, 8: 2048},
            "dit_s": {64: 128, 32: 512, 16: 2048},
            "dit_xl": {64: 64, 32: 256, 16: 1024, 8: 4096, 4: 16384},
            "sit_xl": {32: 256, 16: 1024},
            "sana_cls_xl": {16: 512},
        }
        for model_name, model_cfg in model_cfgs.items():
            input_name, backbone_name = model_name.split("@")
            model = model_cfg.model
            onnx_trt_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(cfg.onnx_trt),
                    OmegaConf.to_container(model_cfg),
                    {
                        "batch_size": num_tokens_to_batch_size[backbone_name][
                            model_cfg[model]["input_size"] // model_cfg[model]["patch_size"]
                        ],
                    },
                )
            )
            print(f"{model_name} inference throughput: {test_onnx_trt(onnx_trt_cfg)}")

    if cfg.test_training_memory:
        for model_name, model_cfg in model_cfgs.items():
            input_name, backbone_name = model_name.split("@")
            training_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(cfg.training),
                    OmegaConf.to_container(OmegaConf.masked_copy(model_cfg, cfg.training.keys())),
                    {"batch_size": 256, "warmup_iterations": 5, "iterations": 10},
                )
            )
            print(f"{model_name} training memory: {test_diffusion_training(training_cfg)}")

    if cfg.test_training_throughput:
        num_tokens_to_batch_size = {
            "uvit_s": {64: 256, 32: 1024, 16: 4096, 8: 16384},
            "uvit_h": {32: 256, 16: 1024, 8: 4096},
            "uvit_2b": {32: 128, 16: 512, 8: 2048},
            "usit_h": {32: 256, 16: 1024, 8: 4096},
            "usit_2b": {32: 128, 16: 512, 8: 2048},
            "usit_3b": {8: 1024},
            "usit_5b": {8: 512},
            "dit_s": {64: 256, 32: 1024, 16: 4096},
            "dit_xl": {64: 64, 32: 256, 16: 1024, 8: 4096, 4: 8192},
            "sit_xl": {32: 256, 16: 1024, 8: 4096},
            "sana_cls_xl": {16: 512},
        }
        for model_name, model_cfg in model_cfgs.items():
            input_name, backbone_name = model_name.split("@")
            model = model_cfg.model
            training_cfg = OmegaConf.to_object(
                OmegaConf.merge(
                    OmegaConf.structured(cfg.training),
                    OmegaConf.to_container(OmegaConf.masked_copy(model_cfg, cfg.training.keys())),
                    {
                        "batch_size": num_tokens_to_batch_size[backbone_name][
                            model_cfg[model]["input_size"] // model_cfg[model]["patch_size"]
                        ]
                    },
                )
            )
            if is_dist_initialized() and training_cfg.fsdp:
                training_throughput = test_diffusion_training(training_cfg)
                if is_master():
                    print(f"{model_name} training throughput: {training_throughput}")
            elif not is_dist_initialized() and not training_cfg.fsdp:
                print(f"{model_name} training throughput: {test_diffusion_training(training_cfg)}")

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
