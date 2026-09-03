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

import imageio
import torch
import torchvision.transforms as transforms
from omegaconf import MISSING, OmegaConf
from PIL import Image

from .dc_ae_v import DCAEV, dc_ae_v_f32t4_chunk_causal, dc_ae_v_f64t4_chunk_causal


@dataclass
class DemoDCAEVModelConfig:
    model_name: str = MISSING
    model_path: str = MISSING
    run_dir: str = MISSING
    input_path_list: tuple[str, ...] = ()
    h: Optional[int] = None
    w: Optional[int] = None
    t: Optional[int] = None


def _resize_center_crop(image: Image.Image, height: int, width: int) -> Image.Image:
    """Resize an RGB image while preserving aspect ratio, then center crop to ``height x width``."""
    if image.size == (width, height):
        return image

    while image.width >= 2 * width and image.height >= 2 * height:
        image = image.resize((image.width // 2, image.height // 2), resample=Image.Resampling.BOX)

    scale = max(height / image.height, width / image.width)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        resample=Image.Resampling.BICUBIC,
    )
    crop_top = (image.height - height) // 2
    crop_left = (image.width - width) // 2
    return image.crop((crop_left, crop_top, crop_left + width, crop_top + height))


@torch.inference_mode()
def main() -> None:
    cfg: DemoDCAEVModelConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(DemoDCAEVModelConfig), OmegaConf.from_cli())
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    if cfg.model_name in [
        "dc-ae-v-1.0-f32t4c32-bf16",
        "dc-ae-v-1.0-f32t4c64-bf16",
        "dc-ae-v-1.0-f32t4c128-bf16",
        "dc-ae-v-1.0-f32t4c256-bf16",
    ]:
        dc_ae_v_config = dc_ae_v_f32t4_chunk_causal(cfg.model_name, cfg.model_path)
    elif cfg.model_name in [
        "dc-ae-v-1.0-f64t4c128-bf16",
    ]:
        dc_ae_v_config = dc_ae_v_f64t4_chunk_causal(cfg.model_name, cfg.model_path)
    else:
        raise ValueError(f"model {cfg.model_name} is not supported")

    dc_ae_v = DCAEV(dc_ae_v_config).to(device=device, dtype=dtype).eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    os.makedirs(cfg.run_dir, exist_ok=True)

    for input_path in cfg.input_path_list:
        if not input_path.endswith(".mp4"):
            raise ValueError(f"Only MP4 inputs are supported, got {input_path!r}")

        video_reader = imageio.get_reader(input_path, format="ffmpeg")
        try:
            frame_count = video_reader.count_frames()
            video_w, video_h = Image.fromarray(video_reader.get_data(0)).size

            w = cfg.w or video_w // dc_ae_v.spatial_compression_ratio * dc_ae_v.spatial_compression_ratio
            h = cfg.h or video_h // dc_ae_v.spatial_compression_ratio * dc_ae_v.spatial_compression_ratio
            t = cfg.t or frame_count // dc_ae_v.temporal_compression_ratio * dc_ae_v.temporal_compression_ratio

            center_frame = frame_count // 2
            frame_indices = range(center_frame - t // 2, center_frame + t // 2 + t % 2)
            frames = [
                _resize_center_crop(
                    Image.fromarray(video_reader.get_data(min(max(frame_index, 0), frame_count - 1))),
                    h,
                    w,
                )
                for frame_index in frame_indices
            ]
        finally:
            video_reader.close()

        video = torch.stack([transform(frame) for frame in frames], dim=1)[None].to(device=device, dtype=dtype)

        latent = dc_ae_v.encode(video)
        y = dc_ae_v.decode(latent)

        output_path = os.path.join(cfg.run_dir, os.path.basename(input_path))
        output_frames = (255 * (y[0].cpu() * 0.5 + 0.5) + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
        video_writer = imageio.get_writer(output_path, mode="I", fps=8, format="mp4", codec="h264")
        try:
            for frame in output_frames:
                video_writer.append_data(frame)
        finally:
            video_writer.close()


if __name__ == "__main__":
    main()
