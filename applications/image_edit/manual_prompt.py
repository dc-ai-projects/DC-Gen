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
from typing import Any, Optional

import torch
from PIL import Image
from tqdm import tqdm

from dc_ai.apps.utils.config import get_config
from dc_ai.apps.utils.dtype import get_dtype_from_str
from dc_ai.imageeditcore.models.base import BaseImageEditModel
from dc_ai.imageeditcore.trainer import (
    ImageEditCoreTrainer,
    ImageEditCoreTrainerConfig,
)


@dataclass
class ImageEditingConfig(ImageEditCoreTrainerConfig):
    eval_data_providers: tuple[str, ...] = ()

    prompts: Any = ("Change the rabbit's color to purple, with a flash light background.",)

    image_paths: Optional[tuple[str, ...]] = ("assets/fig/image_edit_input.jpg",)

    batch_size: int = 4
    num_samples_per_prompt: int = 1

    noise_path: Optional[str] = None


def generate_images(cfg: ImageEditingConfig):
    if isinstance(cfg.prompts, str):
        cfg.prompts = [cfg.prompts]
    prompts = [prompt for _ in range(cfg.num_samples_per_prompt) for prompt in cfg.prompts]
    num_samples = len(prompts)

    image_paths = cfg.image_paths
    assert len(image_paths) == len(cfg.prompts)
    image_paths = [image_path for _ in range(cfg.num_samples_per_prompt) for image_path in image_paths]

    trainer = ImageEditCoreTrainer(cfg)
    network: BaseImageEditModel = trainer.network.eval()

    eval_generator = torch.Generator(device=torch.device("cuda"))
    eval_generator.manual_seed(cfg.seed + trainer.rank)

    trainer.text_encoder = trainer.text_encoder.to(trainer.device)
    trainer.image_encoder = trainer.image_encoder.to(trainer.device)

    text_embed_info_list, neg_text_embed_info_list, image_embed_info_list = [], [], []
    for i in range(0, num_samples, cfg.batch_size):
        images = [
            trainer.image_encoder.spatial_transform(
                Image.open(image_path),
                trainer.autoencoder.spatial_compression_ratio * trainer.spatial_patch_size,
            )
            for image_path in image_paths[i : min(i + cfg.batch_size, num_samples)]
        ]
        images = torch.stack(images, dim=0)
        text_embed_info = trainer.text_encoder.get_text_embed_info(
            prompts=prompts[i : min(i + cfg.batch_size, num_samples)],
            images=images,
            device=trainer.device,
        )
        text_embed_info_list.append(text_embed_info)

        if trainer.cfg.use_neg_prompt:
            neg_text_embed_info = trainer.text_encoder.get_text_embed_info(
                prompts=[" " for _ in range(i, min(i + cfg.batch_size, num_samples))],
                images=images,
                device=trainer.device,
            )
            neg_text_embed_info_list.append(neg_text_embed_info)
        else:
            neg_text_embed_info_list.append(None)

        image_embed_info = trainer.image_encoder.get_image_embed_info(
            images=images,
            autoencoder=trainer.autoencoder,
        )

        image_embed_info_list.append(image_embed_info)

    os.makedirs(cfg.run_dir, exist_ok=True)
    idx = 0

    for text_embed_info, neg_text_embed_info, image_embed_info in zip(
        text_embed_info_list, neg_text_embed_info_list, image_embed_info_list
    ):
        with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=trainer.enable_amp):
            latent_samples = network.generate(
                text_embed_info=text_embed_info,
                neg_text_embed_info=neg_text_embed_info,
                image_embed_info=image_embed_info,
                cfg_scale=cfg.cfg_scale,
                generator=eval_generator,
            )

        image_samples = trainer.autoencoder.decode(latent_samples.to(get_dtype_from_str(cfg.autoencoder_dtype)))
        images_tensor = (image_samples * 0.5 + 0.5).clamp(0, 1)
        images_numpy = images_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
        images_numpy_uint8 = (images_numpy * 255).round().astype("uint8")
        images_PIL = [Image.fromarray(image) for image in images_numpy_uint8]

        for image_idx, image_sample in enumerate(images_PIL):
            save_path = os.path.join(cfg.run_dir, f"inference_{str(idx).zfill(4)}_{str(image_idx).zfill(4)}.jpg")
            image_sample.save(save_path)
        idx += 1

    print(f"Images saved at {cfg.run_dir}.")


def main():
    cfg = get_config(ImageEditingConfig)
    generate_images(cfg)


if __name__ == "__main__":
    main()
