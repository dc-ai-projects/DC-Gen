import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from ...models.utils import get_submodule_weights
from .base import BasePatchEmbedding, BasePatchEmbeddingConfig

__all__ = ["WanT2VPatchEmbedding", "WanI2VPatchEmbedding"]


@dataclass
class WanT2VPatchEmbeddingConfig(BasePatchEmbeddingConfig):
    input_size: tuple[int, int, int] = (21, 60, 104)
    patch_size: tuple[int, int, int] = (1, 2, 2)


@dataclass
class WanI2VPatchEmbeddingConfig(WanT2VPatchEmbeddingConfig):
    t_ratio: int = 4
    i2v_concat: bool = True


class WanT2VPatchEmbedding(BasePatchEmbedding):
    def __init__(self, cfg: WanT2VPatchEmbeddingConfig):
        super().__init__(cfg)
        self.cfg: WanT2VPatchEmbeddingConfig

    def _build_model(self) -> None:
        self.patch_embedding = nn.Conv3d(
            self.cfg.in_channels,
            self.cfg.hidden_size,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["patch_embedding"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self) -> None:
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "dc_adapt":
            if "ema_model_state_dict" in checkpoint:
                checkpoint = list(checkpoint["ema_model_state_dict"].values())[0]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.get_trainable_modules_list().load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "wan":
            self.patch_embedding.load_state_dict(get_submodule_weights(checkpoint, "patch_embedding."), strict=True)
        elif self.cfg.pretrained_source == "dc-ae":
            checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
            target_state_dict = self.get_trainable_modules_list().state_dict()
            for key in target_state_dict:
                target_state_dict[key] = checkpoint[key]
            self.get_trainable_modules_list().load_state_dict(target_state_dict)
        else:
            raise ValueError(f"Unsupported pretrained source {self.cfg.pretrained_source}")

    def initialize_weights(self) -> None:
        # Initialize patch_embed
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        fan_in = self.cfg.in_channels * math.prod(self.cfg.patch_size)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.patch_embedding.bias, -bound, bound)
        self.patch_embedding.weight.initialized = True
        self.patch_embedding.bias.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)
        return x


class WanI2VPatchEmbedding(WanT2VPatchEmbedding):
    def __init__(self, cfg: WanI2VPatchEmbeddingConfig):
        super().__init__(cfg)
        self.cfg: WanI2VPatchEmbeddingConfig

    def _build_model(self) -> None:
        self.patch_embedding = nn.Conv3d(
            self.cfg.in_channels * 2 + self.cfg.t_ratio if self.cfg.i2v_concat else self.cfg.in_channels,
            self.cfg.hidden_size,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )
