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
from functools import partial
from typing import Optional

import ipdb
import torch
import torchvision.transforms as transforms
from omegaconf import MISSING
from PIL import Image
from torch import nn
from torchvision.utils import save_image

from ...ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL
from ...apps.utils.config import get_config
from ...apps.utils.dist import dist_init, get_dist_local_rank
from ...apps.utils.dtype import get_dtype_from_str
from ...apps.utils.efficiency import test_pytorch_efficiency
from ...apps.utils.export import export_onnx
from ...apps.utils.image import DMCrop
from ...apps.utils.tensorrt import get_tensorrt_cmd, get_tensorrt_result, load_tensorrt_result
from ...models.utils.network import get_params_num
from ..models.base import BaseAE, BaseAEConfig
from ..models.dc_ae import DCAE, DCAEConfig


@dataclass
class TestAEEfficiencyConfig:
    model: str = MISSING
    dc_ae: DCAEConfig = field(default_factory=DCAEConfig)

    task: str = "torch_inference"
    get_tensorrt_cmd: bool = False

    warmup_iterations: int = 20
    iterations: int = 100

    dtype: str = "bf16"
    input_shape: tuple[int, int, int, int] = MISSING
    latent_shape: Optional[tuple[int, int, int, int]] = None
    latent_dtype: Optional[str] = None
    run_dir: str = MISSING


def main():
    cfg = get_config(TestAEEfficiencyConfig)
    os.makedirs(cfg.run_dir, exist_ok=True)
    dist_init()
    torch.cuda.set_device(get_dist_local_rank())
    device = torch.device("cuda")
    dtype = get_dtype_from_str(cfg.dtype)
    if cfg.model == "dc_ae":
        model = DCAE(cfg.dc_ae)
    elif cfg.model in REGISTERED_DCAE_MODEL:
        model_cfg_func, pretrained_path, organization = REGISTERED_DCAE_MODEL[cfg.model]
        if pretrained_path is None:
            model = DCAE_HF.from_pretrained(f"{organization}/{cfg.model}")
        else:
            cfg = model_cfg_func(cfg.model, pretrained_path)
            model = DCAE(cfg)
    else:
        raise NotImplementedError(f"model {cfg.model} is not supported")

    model = model.eval().to(device=device)

    transform = transforms.Compose(
        [
            DMCrop(size=(cfg.input_shape[2], cfg.input_shape[3])),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    x = Image.open("assets/fig/girl.png")
    x: torch.Tensor = transform(x)[None].to(device=device)
    with torch.no_grad():
        latent = model.encode(x)
        y = model.decode(latent)
    save_image(torch.cat([x, y], dim=3) * 0.5 + 0.5, os.path.join(cfg.run_dir, "recon.jpg"))
    if cfg.latent_shape is not None:
        latent_shape = cfg.latent_shape
    else:
        latent_shape = (cfg.input_shape[0],) + latent.shape[1:]

    if cfg.task == "torch_inference":
        model = model.to(dtype=dtype)
        x = torch.randn(*cfg.input_shape, device=device, dtype=dtype)
        encode_efficiency = test_pytorch_efficiency(partial(model.encode, x))
        print(f"encode torch inference:")
        print(f"step_time: {encode_efficiency['step_time']}")
        print(f"throughput: {cfg.input_shape[0] * encode_efficiency['throughput']}")
        print(f"memory: {encode_efficiency['memory']}")

        latent = torch.randn(
            *latent_shape,
            device=device,
            dtype=dtype if cfg.latent_dtype is None else get_dtype_from_str(cfg.latent_dtype),
        )
        decode_efficiency = test_pytorch_efficiency(partial(model.decode, latent))
        print(f"decode torch inference:")
        print(f"step_time: {decode_efficiency['step_time']}")
        print(f"throughput: {cfg.input_shape[0] * decode_efficiency['throughput']}")
        print(f"memory: {decode_efficiency['memory']}")
    elif cfg.task == "trt_inference":
        x = torch.randn(*cfg.input_shape, device=device)
        encoder_export_path = os.path.join(cfg.run_dir, "encoder.onnx")
        if not os.path.exists(encoder_export_path):
            export_onnx(model.encoder, encoder_export_path, x, simplify=True, opset=17, large=False)
        result_path = os.path.join(cfg.run_dir, "encoder_trt.txt")
        result = get_tensorrt_result(encoder_export_path, result_path)
        result["throughput"] *= cfg.input_shape[0]
        print(f"encode trt inference: {result}")

        latent_dtype = torch.float if cfg.latent_dtype is None else get_dtype_from_str(cfg.latent_dtype)
        latent = torch.randn(*latent_shape, device=device, dtype=latent_dtype)
        decoder_export_path = os.path.join(cfg.run_dir, "decoder.onnx")
        if not os.path.exists(decoder_export_path):
            export_onnx(model.decoder, decoder_export_path, latent, simplify=True, opset=17, large=False)
        result_path = os.path.join(cfg.run_dir, "decoder_trt.txt")
        result = get_tensorrt_result(decoder_export_path, result_path)
        result["throughput"] *= cfg.input_shape[0]
        print(f"decode trt inference: {result}")
    elif cfg.task == "blockwise_trt_inference":
        torch.set_grad_enabled(False)

        export_path_set: set[str] = set()
        export_list: list[tuple[str, str]] = []

        def export_and_forward(
            module: nn.Module, x: torch.Tensor, module_name: str, module_position: str
        ) -> torch.Tensor:
            y = module(x)
            export_path = os.path.join(
                cfg.run_dir, f"{module_name}_{'_'.join(map(str, x.shape))}__{'_'.join(map(str, y.shape))}.onnx"
            )
            if not os.path.exists(export_path):
                export_onnx(module, export_path, x, simplify=True, opset=17, large=False)
            export_path_set.add(export_path)
            export_list.append((module_position, export_path))
            return y

        x = x.repeat(cfg.input_shape[0], 1, 1, 1)

        x = export_and_forward(model.encoder.project_in, x, "encoder_project_in", "encoder.project_in")
        for stage_id, stage in enumerate(model.encoder.stages):
            stage_block_type = (
                model.cfg.encoder.block_type
                if isinstance(model.cfg.encoder.block_type, str)
                else model.cfg.encoder.block_type[stage_id]
            )
            depth = model.cfg.encoder.depth_list[stage_id]
            for block_id in range(depth):
                block_type = stage_block_type if isinstance(stage_block_type, str) else stage_block_type[block_id]
                x = export_and_forward(
                    stage.op_list[block_id], x, block_type, f"encoder.stage_{stage_id}.block_{block_id}"
                )
            if stage_id < model.encoder.num_stages - 1 and depth > 0:
                x = export_and_forward(
                    stage.op_list[-1],
                    x,
                    model.cfg.encoder.downsample_block_type,
                    f"encoder.stage_{stage_id}.downsample",
                )
        x = export_and_forward(model.encoder.project_out, x, "encoder_project_out", "encoder.project_out")

        x = export_and_forward(model.decoder.project_in, x, "decoder_project_in", "decoder.project_in")
        for stage_id, stage in reversed(list(enumerate(model.decoder.stages))):
            stage_block_type = (
                model.cfg.decoder.block_type
                if isinstance(model.cfg.decoder.block_type, str)
                else model.cfg.decoder.block_type[stage_id]
            )
            depth = model.cfg.decoder.depth_list[stage_id]
            if stage_id < model.decoder.num_stages - 1 and depth > 0:
                x = export_and_forward(
                    stage.op_list[0], x, model.cfg.decoder.upsample_block_type, f"decoder.stage_{stage_id}.upsample"
                )
                main_block_start_id = 1
            else:
                main_block_start_id = 0
            for block_id in range(depth):
                block_type = stage_block_type if isinstance(stage_block_type, str) else stage_block_type[block_id]
                x = export_and_forward(
                    stage.op_list[block_id + main_block_start_id],
                    x,
                    block_type,
                    f"decoder.stage_{stage_id}.block_{block_id}",
                )
        x = export_and_forward(model.decoder.project_out, x, "decoder_project_out", "decoder.project_out")

        save_image(x * 0.5 + 0.5, os.path.join(cfg.run_dir, "recon.jpg"))

        print(export_path_set)
        for export_path in export_path_set:
            result_path = export_path.removesuffix(".onnx") + "_trt.txt"
            if cfg.get_tensorrt_cmd:
                if not os.path.exists(result_path) or len(load_tensorrt_result(result_path)) == 0:
                    print(get_tensorrt_cmd(export_path, result_path))
            else:
                result = get_tensorrt_result(export_path, result_path)

        if not cfg.get_tensorrt_cmd:
            sum_latency: dict[str, float] = {}
            for module_position, export_path in export_list:
                result_path = export_path.removesuffix(".onnx") + "_trt.txt"
                result = get_tensorrt_result(export_path, result_path)
                latency = result["mean_latency"]
                module_position_splitted = module_position.split(".")
                for i in range(len(module_position_splitted)):
                    module_position_prefix = ".".join(module_position_splitted[: i + 1])
                    if module_position_prefix not in sum_latency:
                        sum_latency[module_position_prefix] = 0.0
                    sum_latency[module_position_prefix] += latency

            for key, value in sorted(sum_latency.items()):
                print(f"{key} latency: {value}")
    else:
        raise ValueError(f"task {cfg.task} is not supported")


if __name__ == "__main__":
    main()
