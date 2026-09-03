# Modified from ``https://github.com/QwenLM/Qwen-Image''.
# Qwen-Image-Edit was introduced by Qwen Team.
# The original implementation is by Alibaba Cloud, licensed under the Apache License 2.0.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..aecore.autoencoder import Autoencoder
from ..apps.utils.dtype import get_dtype
from ..apps.utils.image import QwenImageResize
from ..models.utils.network import get_device


@dataclass
class ImageEditCoreImageEncoderConfig:
    # basic
    name: Optional[str] = None

    # transform
    resolution: int = 1024
    spatial_patch_size: Optional[int] = None
    img_preprocess: str = "resize"


class ImageEditCoreImageEncoder(nn.Module):
    def __init__(self, cfg: ImageEditCoreImageEncoderConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.resolution is not None

        if cfg.img_preprocess == "resize":
            self.spatial_transform = QwenImageResize(
                resolution=self.cfg.resolution,
            )
        else:
            raise ValueError(f"Img preprocess {cfg.img_preprocess} is not supported")

    @torch.no_grad()
    def get_ae_feature(
        self,
        images: torch.Tensor,
        autoencoder: Autoencoder,
    ):
        device = get_device(autoencoder)
        dtype = get_dtype(autoencoder)
        images = images.to(device=device, dtype=dtype)
        image_feature = autoencoder.encode(images)

        return image_feature

    @torch.no_grad()
    def get_image_embed_info(
        self,
        images: torch.Tensor,
        autoencoder: Autoencoder,
    ):
        image_embed_info = {}
        image_embed_info["ae_feature"] = self.get_ae_feature(images, autoencoder)
        return image_embed_info
