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
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ...apps.utils.config import get_config
from ...apps.utils.dtype import get_dtype_from_str
from ...apps.utils.efficiency import test_latency
from ..models.base import BaseImageEditModel
from ..trainer import ImageEditCoreTrainer, ImageEditCoreTrainerConfig


@dataclass
class TestInferenceEfficiencyConfig(ImageEditCoreTrainerConfig):
    eval_data_providers: tuple[str, ...] = ()

    prompts: tuple[str, ...] = ("Change the car to a pink one.",)

    image_paths: Optional[tuple[str, ...]] = ("assets/fig/flux_output_3.png",)
    num_samples_per_prompt: int = 1


def main():
    cfg = get_config(TestInferenceEfficiencyConfig)
    latency: dict[str, float] = {}

    if isinstance(cfg.prompts, str):
        cfg.prompts = [cfg.prompts]
    prompts = [prompt for _ in range(cfg.num_samples_per_prompt) for prompt in cfg.prompts]

    image_paths = cfg.image_paths
    assert len(image_paths) == len(cfg.prompts)
    image_paths = [image_path for _ in range(cfg.num_samples_per_prompt) for image_path in image_paths]

    trainer = ImageEditCoreTrainer(cfg)
    network: BaseImageEditModel = trainer.network.eval()

    eval_generator = torch.Generator(device=torch.device("cuda"))
    eval_generator.manual_seed(cfg.seed + trainer.rank)

    trainer.text_encoder = trainer.text_encoder.to(trainer.device)
    trainer.image_encoder = trainer.image_encoder.to(trainer.device)

    images = [
        trainer.image_encoder.spatial_transform(
            Image.open(image_path),
            trainer.autoencoder.spatial_compression_ratio * trainer.spatial_patch_size,
        )
        for image_path in image_paths
    ]
    images = torch.stack(images, dim=0)
    text_embed_info, latency["text_embed_info"] = test_latency(
        trainer.text_encoder.get_text_embed_info,
        prompts=prompts,
        images=images,
        device=trainer.device,
    )

    if trainer.cfg.use_neg_prompt:
        neg_text_embed_info, latency["neg_text_embed_info"] = test_latency(
            trainer.text_encoder.get_text_embed_info,
            prompts=prompts,
            images=images,
            device=trainer.device,
        )
    else:
        neg_text_embed_info = None

    image_embed_info, latency["image_embed_info"] = test_latency(
        trainer.image_encoder.get_image_embed_info,
        images=images,
        autoencoder=trainer.autoencoder,
    )

    os.makedirs(cfg.run_dir, exist_ok=True)

    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=trainer.enable_amp):
        latency["backbone"] = []
        for i in range(5):
            latent_samples, latency_backbone = test_latency(
                network.generate,
                text_embed_info=text_embed_info,
                neg_text_embed_info=neg_text_embed_info,
                image_embed_info=image_embed_info,
                cfg_scale=cfg.cfg_scale,
                generator=eval_generator,
            )
            latency["backbone"].append(latency_backbone)

    image_samples, latency["decode"] = test_latency(
        trainer.autoencoder.decode,
        latent_samples.to(get_dtype_from_str(cfg.autoencoder_dtype)),
    )

    latency["backbone"] = np.mean(latency["backbone"])
    total_latency = sum(value for value in latency.values())
    for key, value in latency.items():
        print(f"{key}: {value:.2f} {value / total_latency:.4f}")


if __name__ == "__main__":
    main()


"""
# Qwen-Image-Edit 512x512
python -m dc_ai.imageeditcore.efficiency.inference \
    model=qwen_image resolution=512 amp=bf16 model_dtype=bf16 \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=512 \
    autoencoder.name=qwen-vae autoencoder_dtype=bf16 \
    qwen_image.in_channels=16 qwen_image.depth=60 qwen_image.hidden_dim=3072 qwen_image.num_heads=24 \
    qwen_image.pretrained_path=assets/checkpoints/image_edit/qwen_image_pretrained.pt qwen_image.pretrained_source=qwen_image \
    cfg_scale=4.0 \
    run_dir=tmp_qwen_image 

"""
