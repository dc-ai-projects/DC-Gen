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
import sys
from dataclasses import dataclass

import numpy as np
import torch
from omegaconf import MISSING
from torchvision.utils import save_image

from dc_gen.apps.utils.config import get_config
from dc_gen.c2i_model_zoo import DCAE_Diffusion_HF
from dc_gen.models.utils.network import get_dtype_from_str


@dataclass
class DemoDiffusionModelConfig:
    model: str = MISSING
    diffusion_model_dtype: str = "fp32"
    autoencoder_dtype: str = "fp32"
    cfg_scale: float = 6.0
    run_dir: str = MISSING
    random_inputs: bool = False
    seed: int = 0


def main():
    torch.set_grad_enabled(False)
    cfg = get_config(DemoDiffusionModelConfig)

    device = torch.device("cuda")
    dc_ae_diffusion = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/{cfg.model}")
    autoencoder = dc_ae_diffusion.autoencoder.eval().to(device=device, dtype=get_dtype_from_str(cfg.autoencoder_dtype))
    diffusion_model = dc_ae_diffusion.diffusion_model.eval().to(
        device=device, dtype=get_dtype_from_str(cfg.diffusion_model_dtype)
    )

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    eval_generator = torch.Generator(device=device)
    eval_generator.manual_seed(cfg.seed)
    if cfg.random_inputs:
        inputs = torch.randint(0, 1000, (16,), device=device)
    else:
        inputs = torch.tensor(
            [279, 333, 979, 936, 933, 145, 497, 1, 248, 360, 793, 12, 387, 437, 938, 978],
            dtype=torch.int,
            device=device,
        )
    num_samples = inputs.shape[0]
    inputs_null = 1000 * torch.ones((num_samples,), dtype=torch.int, device=device)
    latent_samples = diffusion_model.generate(inputs, inputs_null, cfg.cfg_scale, eval_generator)
    latent_samples = latent_samples.to(dtype=get_dtype_from_str(cfg.autoencoder_dtype))
    image_samples = autoencoder.decode(latent_samples)
    save_path = os.path.join(cfg.run_dir, f"demo_random_{cfg.seed}.jpg" if cfg.random_inputs else "demo.jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"saving demo image to {save_path}")
    save_image(image_samples * 0.5 + 0.5, save_path, nrow=int(np.sqrt(num_samples)))


if __name__ == "__main__":
    main()
