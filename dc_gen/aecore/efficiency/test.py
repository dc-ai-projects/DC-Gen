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

import torch
import torchvision.transforms as transforms
from omegaconf import MISSING
from PIL import Image
from torchvision.utils import save_image

from ...ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL
from ...apps.utils.config import get_config
from ...apps.utils.dist import dist_init, get_dist_local_rank
from ...apps.utils.efficiency import test_pytorch_efficiency
from ...apps.utils.export import export_onnx
from ...apps.utils.image import DMCrop
from ...apps.utils.tensorrt import get_tensorrt_result
from ...models.utils.network import get_dtype_from_str, get_params_num
from ..models.base import BaseAE, BaseAEConfig
from ..models.dc_ae import DCAE, DCAEConfig


@dataclass
class TestAEEfficiencyConfig:
    model: str = MISSING
    dc_ae: DCAEConfig = field(default_factory=DCAEConfig)

    task: str = "torch_inference"

    warmup_iterations: int = 20
    iterations: int = 100

    dtype: str = "bf16"
    input_shape: tuple[int, int, int, int] = MISSING
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
    x = transform(x)[None].to(device=device)
    with torch.no_grad():
        latent = model.encode(x)
        y = model.decode(latent)
    save_image(torch.cat([x, y], dim=3) * 0.5 + 0.5, os.path.join(cfg.run_dir, "recon.jpg"))
    latent_shape = (cfg.input_shape[0],) + latent.shape[1:]

    if cfg.task == "torch_inference":
        model = model.to(dtype=dtype)
        x = torch.randn(*cfg.input_shape, device=device, dtype=dtype)
        encode_efficiency = test_pytorch_efficiency(partial(model.encode, x))
        print(f"encode torch inference:")
        print(f"step_time: {encode_efficiency['step_time']}")
        print(f"throughput: {cfg.input_shape[0] * encode_efficiency['throughput']}")
        print(f"memory: {encode_efficiency['memory']}")

        latent = torch.randn(*latent_shape, device=device, dtype=dtype)
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

        latent = torch.randn(*latent_shape, device=device)
        decoder_export_path = os.path.join(cfg.run_dir, "decoder.onnx")
        if not os.path.exists(decoder_export_path):
            export_onnx(model.decoder, decoder_export_path, latent, simplify=True, opset=17, large=False)
        result_path = os.path.join(cfg.run_dir, "decoder_trt.txt")
        result = get_tensorrt_result(decoder_export_path, result_path)
        result["throughput"] *= cfg.input_shape[0]
        print(f"decode trt inference: {result}")


if __name__ == "__main__":
    main()
