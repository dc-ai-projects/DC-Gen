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
import time
from dataclasses import dataclass, field
from typing import Union

import torch
from omegaconf import MISSING
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.api import MixedPrecision
from tqdm import tqdm

from ....apps.utils.dist import get_dist_size, is_master
from ....models.utils.network import get_dtype_from_str
from ..models.dit import DiT, DiTConfig
from ..models.sana_cls import SanaCls, SanaClsConfig
from ..models.uvit import UViT, UViTConfig


@dataclass
class TestDiffusionTrainingConfig:
    model: str = MISSING
    dit: DiTConfig = field(default_factory=DiTConfig)
    uvit: UViTConfig = field(default_factory=UViTConfig)
    sana_cls: SanaClsConfig = field(default_factory=SanaClsConfig)
    batch_size: int = 2
    dtype: str = "fp16"
    warmup_iterations: int = 20
    iterations: int = 100
    fsdp: bool = False
    root_dir: str = os.path.join(os.path.dirname(__file__), "training_efficiency")


def get_result_path(cfg: TestDiffusionTrainingConfig):
    if cfg.model == "dit":
        in_channels = cfg.dit.in_channels
        input_size = cfg.dit.input_size
        result_path = os.path.join(
            cfg.root_dir,
            "dit",
            f"d{cfg.dit.depth}_c{cfg.dit.hidden_size}_h{cfg.dit.num_heads}_p{cfg.dit.patch_size}_u{cfg.dit.unconditional}_nc{cfg.dit.num_classes}_{cfg.dtype}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.json",
        )
    elif cfg.model == "uvit":
        in_channels = cfg.uvit.in_channels
        input_size = cfg.uvit.input_size
        result_path = os.path.join(
            cfg.root_dir,
            "uvit",
            f"d{cfg.uvit.depth}_c{cfg.uvit.hidden_size}_h{cfg.uvit.num_heads}_p{cfg.uvit.patch_size}_nc{cfg.uvit.num_classes}_{cfg.dtype}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.json",
        )
    elif cfg.model == "sana_cls":
        in_channels = cfg.sana_cls.in_channels
        input_size = cfg.sana_cls.input_size
        result_path = os.path.join(
            cfg.root_dir,
            "sana_cls",
            f"d{cfg.sana_cls.depth}_c{cfg.sana_cls.hidden_size}_hd{cfg.sana_cls.attention_head_dim}_p{cfg.sana_cls.patch_size}_nc{cfg.sana_cls.num_classes}_{cfg.dtype}_input{cfg.batch_size}x{in_channels}x{input_size}x{input_size}.json",
        )
    else:
        raise NotImplementedError
    return result_path


def train_step(
    diffusion_model: Union[DiT, UViT],
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    x = torch.randn(
        batch_size,
        diffusion_model.cfg.in_channels,
        diffusion_model.cfg.input_size,
        diffusion_model.cfg.input_size,
        dtype=dtype,
        device=device,
    )
    y = torch.randint(0, max(diffusion_model.cfg.num_classes, 1), (batch_size,), device=device)
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        loss, _ = diffusion_model(x, y)
    loss[0].backward()
    optimizer.step()
    diffusion_model.zero_grad()


def test_diffusion_training(cfg: TestDiffusionTrainingConfig):
    torch.cuda.empty_cache()
    result_path = get_result_path(cfg)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    result = {}
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
    else:
        device = torch.device("cuda")
        dtype = get_dtype_from_str(cfg.dtype)
        if cfg.model == "dit":
            model = DiT(cfg.dit).to(device)
        elif cfg.model == "uvit":
            model = UViT(cfg.uvit).to(device)
        elif cfg.model == "sana_cls":
            model = SanaCls(cfg.sana_cls).to(device)
        else:
            raise NotImplementedError
        model.enable_activation_checkpointing("full")
        if cfg.fsdp:
            device_mesh = init_device_mesh("cuda", (get_dist_size(),))
            model = FullyShardedDataParallel(
                model,
                use_orig_params=True,
                device_mesh=device_mesh,
                mixed_precision=MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32),
            )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        try:
            for i in tqdm(range(cfg.warmup_iterations), desc="warm up"):
                train_step(model, optimizer, cfg.batch_size, dtype, device)

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            for i in tqdm(range(cfg.iterations), desc="test"):
                train_step(model, optimizer, cfg.batch_size, dtype, device)
            torch.cuda.synchronize()
            end_time = time.time()

            result["step_time"] = (end_time - start_time) / cfg.iterations
            result["per_sample_time"] = (end_time - start_time) / cfg.iterations / cfg.batch_size
            result["throughput"] = 1 / result["per_sample_time"]
            result["memory"] = torch.cuda.max_memory_allocated() / 1024**3
        except Exception as e:
            print(e)
            result["throughput"] = "OOM"

        if is_master():
            with open(result_path, "w") as f:
                json.dump(result, f)

    return result
