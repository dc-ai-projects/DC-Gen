# Qwen-Image-Edit was introduced by Qwen Team.
# The original implementation is by Alibaba Cloud, licensed under the Apache License 2.0. See https://github.com/QwenLM/Qwen-Image.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ...imageeditcore.models.qwen_image import QwenImageConfig
from ...models.utils import get_submodule_weights

__all__ = ["QwenImagePatchEmbedding"]


class QwenImagePatchEmbedding(nn.Module):
    def __init__(self, cfg: QwenImageConfig):
        super(QwenImagePatchEmbedding, self).__init__()
        self.cfg = cfg

        self._build_model()
        self.initialize_weights()
        if cfg.pretrained_path is not None:
            self.load_model()

    def _build_model(self) -> None:
        self.img_in = nn.Linear(self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size, self.cfg.hidden_dim)

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["img_in"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self) -> None:
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "qwen_image":
            self.img_in.load_state_dict(get_submodule_weights(checkpoint, "img_in."), strict=True)
        else:
            raise ValueError(f"Unsupported pretrained source {self.cfg.pretrained_source}")

    def initialize_weights(self) -> None:
        # Initialize patch_embed
        nn.init.xavier_uniform_(self.img_in.weight)
        nn.init.constant_(self.img_in.bias, 0)
        self.img_in.weight.initialized = True
        self.img_in.bias.initialized = True

    def pack_latents(self, latents, batch_size, num_channels, h, w):
        latents = latents.view(
            batch_size,
            num_channels,
            h // self.cfg.patch_size,
            self.cfg.patch_size,
            w // self.cfg.patch_size,
            self.cfg.patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size,
            (h // self.cfg.patch_size) * (w // self.cfg.patch_size),
            num_channels * (self.cfg.patch_size * self.cfg.patch_size),
        )

        return latents

    def prepare_latents(self, latents: torch.Tensor, image_latents: torch.Tensor):
        latents, image_latents = latents.unsqueeze(2), image_latents.unsqueeze(2)
        batch_size, num_channels, _, h, w = image_latents.shape
        latents = self.pack_latents(
            latents=latents,
            batch_size=batch_size,
            num_channels=num_channels,
            h=h,
            w=w,
        )
        image_latents = self.pack_latents(
            latents=image_latents,
            batch_size=batch_size,
            num_channels=num_channels,
            h=h,
            w=w,
        )

        return latents, image_latents

    def unpack_latents(self, x, height, width):
        bsz, _, _, hidden_dim = x.shape
        x = x.reshape(bsz, 2, height // self.cfg.patch_size, width // self.cfg.patch_size, hidden_dim)
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(bsz, 2 * hidden_dim, height // self.cfg.patch_size, width // self.cfg.patch_size)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent, ae_feature = torch.chunk(x, chunks=2, dim=1)
        H, W = latent.shape[-2], latent.shape[-1]
        latent, ae_feature = self.prepare_latents(latent, ae_feature)
        x = torch.cat((latent, ae_feature), dim=1)
        x = self.img_in(x)
        x = torch.chunk(x, chunks=2, dim=1)
        x = torch.stack(x, dim=1)
        x = self.unpack_latents(x, height=H, width=W)
        return x
