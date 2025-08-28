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
from dataclasses import dataclass

import torch
import torchvision.transforms as transforms
from omegaconf import MISSING, OmegaConf
from PIL import Image
from torchvision.utils import save_image

from dc_gen.ae_model_zoo import DCAE_HF


@dataclass
class DemoDCAEModelConfig:
    model: str = MISSING
    run_dir: str = MISSING
    input_path_list: tuple[str] = MISSING


def main():
    torch.set_grad_enabled(False)
    cfg: DemoDCAEModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DemoDCAEModelConfig), OmegaConf.from_cli())
    )

    device = torch.device("cuda")
    dc_ae = DCAE_HF.from_pretrained(cfg.model).to(device).eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    os.makedirs(cfg.run_dir, exist_ok=True)

    for input_path in cfg.input_path_list:
        image = Image.open(input_path)
        target_w, target_h = (
            image.size[0] // dc_ae.spatial_compression_ratio * dc_ae.spatial_compression_ratio,
            image.size[1] // dc_ae.spatial_compression_ratio * dc_ae.spatial_compression_ratio,
        )
        image = image.crop((0, 0, target_w, target_h))
        x = transform(image)[None].to(device)
        latent = dc_ae.encode(x)
        y = dc_ae.decode(latent)
        save_image(torch.cat([x, y], dim=3) * 0.5 + 0.5, os.path.join(cfg.run_dir, os.path.basename(input_path)))


if __name__ == "__main__":
    main()

"""
python -m applications.dc_ae.demo_dc_ae_model model=mit-han-lab/dc-ae-f64c128-in-1.0 run_dir=demo/dc-ae-f64c128-in-1.0 input_path_list=[assets/fig/girl.png]

python -m applications.dc_ae.demo_dc_ae_model model=mit-han-lab/dc-ae-f32c32-in-1.0 run_dir=demo/dc-ae-f32c32-in-1.0 input_path_list=[/home/junyuc/dataset/RUGD/RUGD_sample-data/images/creek_00001.png,/home/junyuc/dataset/RUGD/RUGD_sample-data/images/park-1_00001.png,/home/junyuc/dataset/RUGD/RUGD_sample-data/images/trail_00001.png,/home/junyuc/dataset/RUGD/RUGD_sample-data/images/village_00003.png]
"""
