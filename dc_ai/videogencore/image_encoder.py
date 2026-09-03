# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from ..apps.utils.dtype import get_dtype, get_dtype_from_str
from ..apps.utils.image import WanAspectRatioResize, WanAspectRatioResizeCrop
from ..models.utils.network import get_device
from ..staecore.autoencoder import Autoencoder, AutoencoderConfig


@dataclass
class ImageEncoderConfig:
    # basic
    use_vlm: bool = True
    vlm_name: str = "openai/clip-vit-huge-patch14"
    vlm_dtype: str = "fp16"
    vlm_backbone_path: str = "assets/checkpoints/i2v/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

    # encode
    use_mask: bool = True

    # transform
    resolution: Optional[str] = None
    img_preprocess: str = "resize_crop"
    padding_mode: str = "zeros"  # "zeros" | "replicate"


class ImageEncoder(nn.Module):
    def __init__(self, cfg: ImageEncoderConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.vlm_dtype = get_dtype_from_str(cfg.vlm_dtype)

        if cfg.use_vlm:
            if cfg.vlm_name in ["openai/clip-vit-huge-patch14"]:
                from .models.wan_blocks.clip import CLIPModel

                self.vlm = CLIPModel(
                    dtype=self.vlm_dtype,
                    device=device,
                    checkpoint_path=cfg.vlm_backbone_path,
                )
            else:
                raise ValueError(f"Image encoder {cfg.vlm_name} is not supported")

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        assert cfg.resolution is not None

        if cfg.img_preprocess == "resize":
            self.spatial_transform = WanAspectRatioResize(resolution=self.cfg.resolution)
        elif cfg.img_preprocess == "resize_crop":
            self.spatial_transform = WanAspectRatioResizeCrop(resolution=self.cfg.resolution)
        else:
            raise ValueError(f"Img preprocess {cfg.img_preprocess} is not supported")

    def get_num_frames(self, autoencoder_name: str) -> int:
        if autoencoder_name in [
            "wan-vae",
        ]:
            return 81
        elif autoencoder_name in [
            "wan-2.2-vae",
        ]:
            return 1
        elif autoencoder_name in [
            "dc-ae-v-1.0-f32t4c32-bf16",
            "dc-ae-v-1.0-f32t4c32-bf16-spatial-tile-512",
            "dc-ae-v-1.0-f64t4c128-bf16",
        ]:
            return 80
        else:
            raise ValueError(f"autoencoder {autoencoder_name} is not supported")

    def get_image_mask(self, image_feature, autoencoder: Autoencoder):
        """
        image_feature: (B, C, T, H, W)
        """
        bs, _, num_frames, lat_h, lat_w = image_feature.shape
        device = image_feature.device

        if autoencoder.cfg.name in [
            "wan-vae",
            "wan-2.2-vae",
            "dc-ae-v-1.0-f32t4c32-bf16",
            "dc-ae-v-1.0-f32t4c32-bf16-spatial-tile-512",
            "dc-ae-v-1.0-f64t4c128-bf16",
        ]:
            msk = torch.ones(num_frames * 4, lat_h, lat_w, device=device)
            msk[4:] = 0
        else:
            raise ValueError(f"autoencoder {autoencoder.cfg.name} is not supported")

        t_ratio = autoencoder.temporal_compression_ratio
        msk = msk.view(msk.shape[0] // t_ratio, t_ratio, lat_h, lat_w)
        msk = msk.transpose(0, 1)
        msk = msk.repeat(bs, 1, 1, 1, 1)
        return msk

    @torch.no_grad()
    def get_ae_feature(
        self,
        img: Image.Image | torch.Tensor,
        autoencoder: Autoencoder,
        spatial_patch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):  # Get Image Feature for Single Image or Tensor of Images
        """
        img: PIL Image or Tensor of shape (B, C, H, W) and range [-1, 1]
        """
        device = get_device(autoencoder)
        dtype = get_dtype(autoencoder)
        if isinstance(img, Image.Image):
            if self.cfg.img_preprocess == "resize":
                assert isinstance(spatial_patch_size, int)
                img_tensor = self.spatial_transform(
                    img, total_spatial_compression_ratio=autoencoder.spatial_compression_ratio * spatial_patch_size
                )
            elif self.cfg.img_preprocess == "resize_crop":
                assert isinstance(spatial_patch_size, int)
                img = self.spatial_transform(
                    img, total_spatial_compression_ratio=autoencoder.spatial_compression_ratio * spatial_patch_size
                )
                img_tensor = self.transform(img)[None]
            else:
                raise ValueError(f"Image preprocessing way {self.cfg.img_preprocess} is not suppoted.")
        elif isinstance(img, torch.Tensor):
            img_tensor = img
        else:
            raise ValueError(f"img type {type(img)} not supported")
        autoencoder_name = autoencoder.cfg.name
        num_frames = num_frames or self.get_num_frames(autoencoder_name)
        bs, _, h, w = img_tensor.shape

        img_tensor = img_tensor[:, :, None].to(device=device, dtype=dtype)  # (B, C, 1, H, W)

        if self.cfg.padding_mode == "replicate":
            # Replicate the conditioning frame across the time dimension.
            img_tensor = img_tensor.expand(-1, -1, num_frames, -1, -1).contiguous()
        elif self.cfg.padding_mode == "zeros":
            if autoencoder_name in [
                "wan-vae",
                "wan-2.2-vae",
            ]:
                img_pad = torch.zeros(bs, 3, num_frames - 1, h, w, device=device, dtype=dtype)
                img_tensor = torch.concat([img_tensor, img_pad], dim=2)
            elif autoencoder_name in [
                "dc-ae-v-1.0-f32t4c32-bf16",
                "dc-ae-v-1.0-f32t4c32-bf16-spatial-tile-512",
                "dc-ae-v-1.0-f64t4c128-bf16",
            ]:
                img_pad = torch.zeros(bs, 3, num_frames - 4, h, w, device=device, dtype=dtype)
                img_tensor = torch.concat([img_tensor, img_tensor, img_tensor, img_tensor, img_pad], dim=2)
            else:
                raise ValueError(f"Unsupported autoencoder type {autoencoder_name}")
        else:
            raise ValueError(f"padding_mode {self.cfg.padding_mode!r} is not supported")

        image_feature = autoencoder.encode(img_tensor)

        if self.cfg.use_mask:
            msk = self.get_image_mask(image_feature, autoencoder)
            return torch.cat((msk, image_feature), dim=1)
        else:
            return image_feature

    def get_vlm_feature(self, img: Image.Image | torch.Tensor):
        """
        img: PIL Image or Tensor of shape (B, C, H, W) and range [-1, 1]
        """
        if isinstance(img, Image.Image):
            img = self.transform(img).to(self.device)
            return self.vlm.visual([img[:, None, :, :]])
        elif isinstance(img, torch.Tensor):
            img = img.transpose(0, 1)  # BCHW -> CBHW
            return self.vlm.visual([img])
        else:
            raise ValueError(f"img type {type(img)} not supported")

    def get_image_embed_info(
        self,
        images: list[Image.Image] | torch.Tensor,
        autoencoder: Autoencoder,
        spatial_patch_size: Optional[int] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        image_embed_info = {}
        if isinstance(images, torch.Tensor):
            image_embed_info["ae_feature"] = self.get_ae_feature(
                images, autoencoder, spatial_patch_size=spatial_patch_size, num_frames=num_frames
            )
        elif isinstance(images, list) and all(isinstance(image, Image.Image) for image in images):
            image_embed_info["ae_feature"] = torch.cat(
                [
                    self.get_ae_feature(
                        image,
                        autoencoder,
                        spatial_patch_size=spatial_patch_size,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                    )
                    for image in images
                ]
            )
        else:
            raise ValueError(f"images type {type(images)} not supported")
        if self.cfg.use_vlm:
            image_embed_info["vlm_feature"] = torch.cat([self.get_vlm_feature(image) for image in images])
        return image_embed_info
