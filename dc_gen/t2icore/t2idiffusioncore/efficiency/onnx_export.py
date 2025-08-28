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

import ipdb
import torch

from ....apps.utils.export import export_onnx
from ..models.base import BaseT2IDiffusionModel
from ..models.sana_t2i import SanaT2I
from .model import ModelConfig


class OnnxDiffusionModel(torch.nn.Module):
    def __init__(self, model: BaseT2IDiffusionModel):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, sample_inputs):
        return self.model.forward_without_cfg(**sample_inputs)


def get_export_path(cfg: ModelConfig):
    root_dir = os.path.join(os.path.dirname(__file__), "onnx")
    if cfg.model == "sana_t2i":
        in_channels = cfg.sana_t2i.in_channels
        input_size = cfg.sana_t2i.input_size
        export_path = os.path.join(
            root_dir,
            "sana_t2i",
            f"d{cfg.sana_t2i.depth}_c{cfg.sana_t2i.hidden_size}_h{cfg.sana_t2i.num_heads}_p{cfg.sana_t2i.patch_size}_qknorm{cfg.sana_t2i.qk_norm}_crossnorm{cfg.sana_t2i.cross_norm}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}",
            "model.onnx",
        )
    else:
        raise NotImplementedError
    return export_path


def onnx_export_diffusion(cfg: ModelConfig):
    export_path = get_export_path(cfg)
    if os.path.exists(export_path):
        return export_path

    device = torch.device("cuda")
    # text_encoder = TextEncoder(cfg.text_encoders).to(device=device)

    if cfg.model == "sana_t2i":
        model = SanaT2I(cfg.sana_t2i).to(device)
        sample_inputs = {
            "x": torch.randn(
                cfg.batch_size,
                cfg.sana_t2i.in_channels,
                cfg.sana_t2i.input_size,
                cfg.sana_t2i.input_size,
                device=device,
            ),
            "t": 1000 * torch.rand(cfg.batch_size, device=device),
            "y": torch.randn(
                cfg.batch_size, 1, cfg.sana_t2i.text_max_length, cfg.sana_t2i.caption_channels, device=device
            ),
            "mask": torch.ones((cfg.batch_size, cfg.sana_t2i.text_max_length), device=device, dtype=torch.int64),
        }
    else:
        raise NotImplementedError

    for param in list(model.parameters()):
        torch.nn.init.normal_(param.data, 0.0, 0.02)

    onnx_diffusion_model = OnnxDiffusionModel(model)
    export_onnx(
        onnx_diffusion_model,
        export_path,
        {"sample_inputs": sample_inputs},
        simplify=True,
        opset=cfg.opset,
        large=cfg.large,
    )
    return export_path
