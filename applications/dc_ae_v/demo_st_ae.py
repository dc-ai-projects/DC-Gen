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
from typing import Optional

import ipdb
import torch
import torchvision.transforms as transforms
from omegaconf import MISSING
from PIL import Image
from torchvision.utils import save_image

from dc_ai.apps.utils.config import get_config
from dc_ai.apps.utils.dist import dist_init, distribute_list_to_rank, get_dist_local_rank, is_dist_initialized
from dc_ai.apps.utils.dtype import get_dtype_from_str
from dc_ai.apps.utils.image import DMCrop
from dc_ai.apps.utils.video import VideoLoader, VideoSizeTransform, write_video
from dc_ai.staecore.autoencoder import Autoencoder, AutoencoderConfig
from dc_ai.staecore.models.dc_ae_v import DCAEV, DCAEVConfig


@dataclass
class DemoSTAEModelConfig:
    model: str = "autoencoder"
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    dc_ae_v: DCAEVConfig = field(default_factory=DCAEVConfig)
    run_dir: str = MISSING

    dtype: str = "bf16"
    input_path_list: Optional[tuple[str, ...]] = None
    input_dir: Optional[str] = None
    size_transform: str = "TemporalCenterCropSpatialDMCropResampleRound"
    h: Optional[int] = None
    w: Optional[int] = None
    t: Optional[int] = None
    fps: Optional[float] = None
    max_h: Optional[int] = None
    max_w: Optional[int] = None
    output_fps: float = 8.0
    model_path: Optional[str] = None
    crop_range: Optional[tuple[float, float, float, float]] = None  # left upper right lower
    latent_noise_std: Optional[float] = None
    save_input_sample: Optional[bool] = True


def main():
    dist_init()
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())

    torch.set_grad_enabled(False)
    cfg = get_config(DemoSTAEModelConfig)

    device = torch.device("cuda")
    dtype = get_dtype_from_str(cfg.dtype)
    if cfg.model == "autoencoder":
        st_ae = Autoencoder(cfg.autoencoder).to(device=device, dtype=dtype)
    elif cfg.model == "dc_ae_v":
        st_ae = DCAEV(cfg.dc_ae_v).to(device=device, dtype=dtype).eval()
    else:
        raise ValueError(f"model {cfg.model} not supported")

    if cfg.size_transform == "TemporalCenterCropSpatialDMCropResampleRound":
        size_transform = VideoSizeTransform("CenterCrop", "DMCrop", "Round")
    elif cfg.size_transform == "TemporalCenterCropSpatialResizeResampleRound":
        size_transform = VideoSizeTransform("CenterCrop", "Resize", "Round")
    elif cfg.size_transform == "TemporalKeepAllSpatialDMCropResampleRound":
        size_transform = VideoSizeTransform("KeepAll", "DMCrop", "Round")
    else:
        raise ValueError(f"size transform {cfg.size_transform} not supported")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    os.makedirs(cfg.run_dir, exist_ok=True)

    if cfg.input_path_list is not None:
        input_path_list = list(cfg.input_path_list)
    elif cfg.input_dir is not None:
        input_path_list = [
            os.path.join(cfg.input_dir, file_name)
            for file_name in sorted(os.listdir(cfg.input_dir))
            if os.path.splitext(file_name)[1] in [".mp4", ".png"]
        ]
    else:
        raise ValueError(f"no input provided")

    input_path_list = distribute_list_to_rank(input_path_list)

    spatial_compression_ratio = st_ae.spatial_compression_ratio
    temporal_compression_ratio = st_ae.temporal_compression_ratio
    temporal_remainder = st_ae.temporal_remainder

    for input_path in input_path_list:
        if input_path.endswith(".mp4"):
            save_filename = os.path.basename(input_path)
            video_loader = VideoLoader(input_path, crop_range=cfg.crop_range)
            video_w, video_h = video_loader.get_frames([0])[0].size

            w = cfg.w or video_w // spatial_compression_ratio * spatial_compression_ratio
            h = cfg.h or video_h // spatial_compression_ratio * spatial_compression_ratio
            if cfg.max_w is not None:
                w = min(w, cfg.max_w)
            if cfg.max_h is not None:
                h = min(h, cfg.max_h)

            frames = size_transform(
                video_loader, video_loader.get_frame_count(), video_loader.get_fps(), h=h, w=w, t=cfg.t, fps=cfg.fps
            )
            video = torch.stack([transform(frame) for frame in frames], dim=1)[None].to(device=device, dtype=dtype)
            print(f"input shape: {video.shape}")

            if cfg.latent_noise_std is None:
                y, info = st_ae.reconstruct_video(video)
                latent = info["latent"]
            else:
                latent = st_ae.encode(video)
                y = st_ae.decode(latent + cfg.latent_noise_std * torch.randn_like(latent))
            print(f"latent shape: {latent.shape}, std: {latent.std()}")

            output_sample = y[0].cpu()
            if cfg.save_input_sample:
                output_sample = torch.cat([video[0].cpu(), output_sample], dim=2)
            write_video(
                os.path.join(cfg.run_dir, save_filename),
                output_sample * 0.5 + 0.5,
                fps=cfg.output_fps,
            )
        elif input_path.endswith(".png"):
            image = Image.open(input_path)
            if cfg.crop_range is not None:
                image = image.crop(cfg.crop_range)
            w = cfg.w or image.size[0] // spatial_compression_ratio * spatial_compression_ratio
            h = cfg.h or image.size[1] // spatial_compression_ratio * spatial_compression_ratio
            image_transform = transforms.Compose(
                [
                    DMCrop(size=(h, w)),
                    transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5),
                ]
            )
            x = image_transform(image)[None].to(device=device, dtype=dtype)
            y, info = st_ae.reconstruct_image(x)
            output_sample = y
            if cfg.save_input_sample:
                output_sample = torch.cat([x, output_sample], dim=2)
            save_image(output_sample * 0.5 + 0.5, os.path.join(cfg.run_dir, os.path.basename(input_path)))
            print(f"latent shape {info['latent'].shape}")
        else:
            raise NotImplementedError(f"input_path {input_path} not supported")


if __name__ == "__main__":
    main()
