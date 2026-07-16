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
from PIL import Image

from dc_gen.apps.utils.config import get_config
from dc_gen.models.utils.network import get_dtype_from_str
from dc_gen.t2icore.t2idiffusioncore.trainer import T2IDiffusionCoreTrainer, T2IDiffusionCoreTrainerConfig


@dataclass
class ImageGenerationConfig(T2IDiffusionCoreTrainerConfig):
    prompts: tuple[str, ...] = (
        "In the warm sunlight, a calico cat backflips over the golden dog.",
        "An elephant holding a pink lighting sign saying 'Hello' ",
        "A cute girl with a deer, cartoon",
        "An astronaut cute panda",
        "A robot watches another play snooker.",
        "Beautiful sunset at the beach",
        "Sunflowers in wind",
        "A blackboard with multiple physics formulas",
    )


def generate_images(cfg: ImageGenerationConfig):
    trainer = T2IDiffusionCoreTrainer(cfg)

    trainer.network.eval()
    eval_generator = torch.Generator(device=torch.device("cuda"))
    eval_generator.manual_seed(cfg.seed + trainer.rank)

    text_embed_info = trainer.text_encoder.get_text_embed_info(cfg.prompts, trainer.device)

    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=trainer.enable_amp):
        latent_samples = trainer.network.generate(
            text_embed_info=text_embed_info,
            cfg_scale=cfg.cfg_scale,
            pag_scale=cfg.pag_scale,
            generator=eval_generator,
        )

    image_samples = trainer.autoencoder.decode(latent_samples.to(get_dtype_from_str(cfg.autoencoder_dtype)))

    if cfg.model in ["flux", "hidream"]:
        image_samples = (image_samples * 0.5 + 0.5).clamp(0, 1)
        image_samples_uint8 = (255.0 * image_samples).round().to(dtype=torch.uint8)
    else:
        image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
    image_samples_numpy = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()
    image_samples_PIL = [Image.fromarray(image) for image in image_samples_numpy]

    os.makedirs(cfg.run_dir, exist_ok=True)
    for idx, img in enumerate(image_samples_PIL):
        img.save(os.path.join(cfg.run_dir, f"inference_{str(idx).zfill(4)}.jpg"))

    print(f"Images saved at {cfg.run_dir}.")


def main():
    cfg = get_config(ImageGenerationConfig)
    generate_images(cfg)


if __name__ == "__main__":
    main()
