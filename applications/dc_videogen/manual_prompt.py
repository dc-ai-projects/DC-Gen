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
from dc_ai.apps.utils.video import write_video
from dc_ai.videogencore.models.base import BaseVideoGenModel
from dc_ai.videogencore.trainer import (
    VideoGenCoreTrainer,
    VideoGenCoreTrainerConfig,
)


@dataclass
class VideoGenerationConfig(VideoGenCoreTrainerConfig):
    eval_data_providers: tuple[str, ...] = ()

    prompts: Any = (
        "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    )

    image_paths: Optional[tuple[str, ...]] = ("assets/fig/wan_i2v_input.JPG",)

    batch_size: int = 4
    num_samples_per_prompt: int = 1

    noise_path: Optional[str] = None
    input_path: Optional[str] = None

    height: Optional[int] = None
    width: Optional[int] = None


def generate_videos(cfg: VideoGenerationConfig):
    if isinstance(cfg.prompts, str):
        cfg.prompts = [cfg.prompts]
    prompts = [prompt for _ in range(cfg.num_samples_per_prompt) for prompt in cfg.prompts]
    num_samples = len(prompts)

    if cfg.image_paths is not None and cfg.model in [
        "wan_i2v",
        "moe_wan_i2v",
    ]:
        image_paths = cfg.image_paths
        assert len(image_paths) == len(cfg.prompts)
        image_paths = [image_path for _ in range(cfg.num_samples_per_prompt) for image_path in image_paths]
    else:
        image_paths = None

    trainer = VideoGenCoreTrainer(cfg)
    network: BaseVideoGenModel = trainer.network.eval()

    eval_generator = torch.Generator(device=torch.device("cuda"))
    eval_generator.manual_seed(cfg.seed + trainer.rank)

    trainer.text_encoder = trainer.text_encoder.to(trainer.device)
    if trainer.cfg.offload and cfg.model in ["wan_i2v", "moe_wan_i2v"]:
        if cfg.image_encoder.use_vlm:
            trainer.image_encoder.vlm.model = trainer.image_encoder.vlm.model.to(trainer.device)
        trainer.autoencoder = trainer.autoencoder.to(trainer.device)

    text_embed_info_list, image_embed_info_list = [], []
    for i in range(0, num_samples, cfg.batch_size):
        text_embed_info = trainer.text_encoder.get_text_embed_info(
            prompts=prompts[i : min(i + cfg.batch_size, num_samples)],
            device=trainer.device,
        )
        text_embed_info_list.append(text_embed_info)

        if image_paths is not None and cfg.model in [
            "wan_i2v",
            "moe_wan_i2v",
        ]:
            images = [Image.open(image_path) for image_path in image_paths[i : min(i + cfg.batch_size, num_samples)]]
            if cfg.model == "wan_i2v":
                patch_size = trainer.cfg.wan_i2v.patch_size[1]
            elif cfg.model == "moe_wan_i2v":
                patch_size = trainer.cfg.moe_wan_i2v.patch_size[1]
            else:
                raise ValueError(f"Model type {cfg.model} is not supported.")
            image_embed_info = trainer.image_encoder.get_image_embed_info(
                images,
                trainer.autoencoder,
                patch_size,
                num_frames=cfg.image_embed_info_num_frames,
                height=cfg.height,
                width=cfg.width,
            )
        else:
            image_embed_info = {}

        image_embed_info_list.append(image_embed_info)

    trainer.text_encoder = trainer.text_encoder.to("cpu")
    if trainer.cfg.offload and cfg.model in ["wan_i2v", "moe_wan_i2v"]:
        if cfg.image_encoder.use_vlm:
            trainer.image_encoder.vlm.model = trainer.image_encoder.vlm.model.to("cpu")
        trainer.autoencoder = trainer.autoencoder.to("cpu")

    os.makedirs(cfg.run_dir, exist_ok=True)
    idx = 0

    if trainer.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
        network.submodel[0] = network.submodel[0].to(trainer.device)
    elif trainer.cfg.model in [
        "wan_t2v",
        "wan_i2v",
    ]:
        network = network.to(trainer.device)
    else:
        raise ValueError(f"Model type {trainer.cfg.model} is not supported.")

    noise = torch.load(cfg.noise_path, map_location="cpu").to(trainer.device) if cfg.noise_path is not None else None

    for text_embed_info, image_embed_info in tqdm(zip(text_embed_info_list, image_embed_info_list)):
        if cfg.input_path is not None:
            input_data = torch.load(cfg.input_path, weights_only=True)
            noise = input_data["latents"]
            text_embed_info[trainer.cfg.text_encoders[0]]["text_embeddings"] = input_data["prompt_embeds"]
            image_embed_info["ae_feature"] = input_data["condition"]

        with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=trainer.enable_amp):
            latent_samples, _ = network.generate(
                text_embed_info=text_embed_info,
                image_embed_info=image_embed_info,
                noise=noise,
                cfg_scale=cfg.cfg_scale,
                pag_scale=cfg.pag_scale,
                generator=eval_generator,
            )  # ([B,C,T,H,W], info)

        if trainer.cfg.offload:
            if trainer.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
                network.submodel[-1] = network.submodel[-1].to("cpu")
            else:
                network = network.to("cpu")

        if trainer.cfg.model == "wan_i2v" and not cfg.wan_i2v.i2v_concat:
            latent_samples[:, :, 0] = image_embed_info["ae_feature"][:, :, 0]
        elif trainer.cfg.model == "moe_wan_i2v" and not cfg.moe_wan_i2v.i2v_concat:
            latent_samples[:, :, 0] = image_embed_info["ae_feature"][:, :, 0]

        trainer.autoencoder = trainer.autoencoder.to(trainer.device)
        video_samples = trainer.autoencoder.decode(latent_samples.to(get_dtype_from_str(cfg.autoencoder_dtype)))
        trainer.autoencoder = trainer.autoencoder.to("cpu")

        for video_sample in video_samples:
            save_path = os.path.join(cfg.run_dir, f"inference_{str(idx).zfill(4)}.mp4")
            write_video(save_path, video_sample * 0.5 + 0.5, fps=16)
            idx += 1

        if trainer.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
            network.submodel[0] = network.submodel[0].to(trainer.device)
        elif trainer.cfg.model in [
            "wan_t2v",
            "wan_i2v",
        ]:
            network = network.to(trainer.device)
        else:
            raise ValueError(f"Model type {trainer.cfg.model} is not supported.")

    print(f"Videos saved at {cfg.run_dir}.")


def main():
    cfg = get_config(VideoGenerationConfig)
    generate_videos(cfg)


if __name__ == "__main__":
    main()
