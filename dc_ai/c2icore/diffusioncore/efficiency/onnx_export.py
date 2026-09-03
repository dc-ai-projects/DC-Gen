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
from dataclasses import dataclass, field

import torch
from omegaconf import MISSING

from ....apps.utils.export import export_onnx
from ..models.base import BaseDiffusionModel
from ..models.dit import DiT, DiTConfig
from ..models.sana_cls import SanaCls, SanaClsConfig
from ..models.uvit import UViT, UViTConfig


@dataclass
class OnnxExportDiffusionConfig:
    model: str = MISSING
    dit: DiTConfig = field(default_factory=DiTConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)
    sana_cls: SanaClsConfig = field(default_factory=SanaClsConfig)
    batch_size: int = 2
    opset: int = 17
    large: bool = False
    root_dir: str = os.path.join(os.path.dirname(__file__), "onnx")


class OnnxDiffusionModel(torch.nn.Module):
    def __init__(self, model: BaseDiffusionModel):
        super().__init__()
        self.model = model

    def forward(self, x, t, y):
        return self.model.forward_without_cfg(x, t, y)[0]


def get_export_path(cfg: OnnxExportDiffusionConfig):
    if cfg.model == "dit":
        in_channels = cfg.dit.in_channels
        input_size = cfg.dit.input_size
        export_path = os.path.join(
            cfg.root_dir,
            "dit",
            f"d{cfg.dit.depth}_c{cfg.dit.hidden_size}_h{cfg.dit.num_heads}_p{cfg.dit.patch_size}_u{cfg.dit.unconditional}_nc{cfg.dit.num_classes}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.onnx",
        )
    elif cfg.model == "uvit":
        in_channels = cfg.uvit.in_channels
        input_size = cfg.uvit.input_size
        export_path = os.path.join(
            cfg.root_dir,
            "uvit",
            f"d{cfg.uvit.depth}_c{cfg.uvit.hidden_size}_h{cfg.uvit.num_heads}_p{cfg.uvit.patch_size}_nc{cfg.uvit.num_classes}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.onnx",
        )
    elif cfg.model == "sana_cls":
        in_channels = cfg.sana_cls.in_channels
        input_size = cfg.sana_cls.input_size
        export_path = os.path.join(
            cfg.root_dir,
            "sana_cls",
            f"d{cfg.sana_cls.depth}_c{cfg.sana_cls.hidden_size}_hd{cfg.sana_cls.attention_head_dim}_p{cfg.sana_cls.patch_size}_nc{cfg.sana_cls.num_classes}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.onnx",
        )
    else:
        raise NotImplementedError
    return export_path


def onnx_export_diffusion(cfg: OnnxExportDiffusionConfig):
    export_path = get_export_path(cfg)
    if os.path.exists(export_path):
        return export_path

    device = torch.device("cuda")

    if cfg.model == "dit":
        model = DiT(cfg.dit).to(device)
        in_channels = cfg.dit.in_channels
        input_size = cfg.dit.input_size
    elif cfg.model == "uvit":
        model = UViT(cfg.uvit).to(device)
        in_channels = cfg.uvit.in_channels
        input_size = cfg.uvit.input_size
    elif cfg.model == "sana_cls":
        model = SanaCls(cfg.sana_cls).to(device)
        in_channels = cfg.sana_cls.in_channels
        input_size = cfg.sana_cls.input_size
    else:
        raise NotImplementedError

    for param in list(model.parameters()):
        torch.nn.init.normal_(param.data, 0.0, 0.02)

    onnx_diffusion_model = OnnxDiffusionModel(model)

    x = torch.randn(cfg.batch_size, in_channels, input_size, input_size, device=device)
    t = torch.rand(cfg.batch_size, device=device)
    y = torch.randint(0, max(model.cfg.num_classes, 1), (cfg.batch_size,), device=device)
    sample_inputs = (x, t, y)
    export_onnx(onnx_diffusion_model, export_path, sample_inputs, simplify=True, opset=cfg.opset, large=cfg.large)

    return export_path
