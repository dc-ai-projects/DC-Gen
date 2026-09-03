# FLUX was introduced by Black Forest Lab.
# The original implementation is by Black Forest Lab, licensed under the Apache License 2.0. See https://github.com/black-forest-labs/flux.

import torch
import torch.nn as nn
from einops import rearrange

from ...models.utils import get_submodule_weights
from ...t2icore.models.flux import FluxConfig

__all__ = ["FluxPatchEmbedding"]


class FluxPatchEmbedding(nn.Module):
    def __init__(self, cfg: FluxConfig):
        super(FluxPatchEmbedding, self).__init__()
        self.cfg = cfg

        self._build_model()
        self.initialize_weights()
        if cfg.pretrained_path is not None:
            self.load_model()

    def _build_model(self) -> None:
        self.x_embedder = nn.Linear(
            self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size, self.cfg.hidden_dim
        )

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in ["x_embedder"]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self) -> None:
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "flux":
            self.x_embedder.load_state_dict(get_submodule_weights(checkpoint, "x_embedder."), strict=True)
        else:
            raise ValueError(f"Unsupported pretrained source {self.cfg.pretrained_source}")

    def initialize_weights(self) -> None:
        # Initialize patch_embed nn.Linear
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        self.x_embedder.weight.initialized = True
        self.x_embedder.bias.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, num_channels, height, width = x.shape

        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=self.cfg.patch_size, p2=self.cfg.patch_size)
        x = self.x_embedder(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=height // self.cfg.patch_size)
        return x
