# DiT was introduced by William Peebles and Saining Xie in "Scalable Diffusion Models with Transformers", see http://arxiv.org/abs/2212.09748.
# The original implementation is by Meta Platforms, Inc. and affiliates, licensed under CC-BY-NC. See https://github.com/facebookresearch/DiT.

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...apps.utils.init import init_modules
from ...models.nn.norm import build_norm
from ...models.nn.ops import GLUMBConv, ResidualBlock
from ...models.utils import get_submodule_weights


@dataclass
class BasePatchEmbeddingConfig:
    in_channels: int = 4
    input_size: int = 32
    patch_size: int = 2
    hidden_size: int = 1152
    mlp_ratio: float = 4.0
    patch_kernel_size: Optional[int] = None
    patch_ffn_depth: int = 0

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae"


class BasePatchEmbedding(nn.Module):
    def __init__(self, cfg: BasePatchEmbeddingConfig):
        super().__init__()
        self.cfg = cfg

        self._build_model()
        self.initialize_weights()
        self.check_initialization()
        if cfg.pretrained_path is not None:
            self.load_model()

    def build_x_embedder(self):
        raise NotImplementedError

    def _build_model(self) -> None:
        self.build_x_embedder()
        if self.cfg.patch_ffn_depth > 0:
            self.patch_ffn = nn.Sequential(
                *[
                    ResidualBlock(
                        main=GLUMBConv(
                            self.cfg.hidden_size,
                            self.cfg.hidden_size,
                            expand_ratio=self.cfg.mlp_ratio,
                            use_bias=True,
                            norm=None,
                            act_func=("silu", "silu", None),
                        ),
                        shortcut=nn.Identity(),
                        pre_norm=build_norm("trms2d", self.cfg.hidden_size),
                    )
                    for _ in range(self.cfg.patch_ffn_depth)
                ]
            )
        else:
            self.patch_ffn = None

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        patch_embedding = {}
        for name, module in self.named_children():
            if name in ["x_embedder", "patch_ffn"]:
                patch_embedding[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        patch_embedding = nn.ModuleDict(patch_embedding)

        trainable_modules_list.append(patch_embedding)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self) -> None:
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "dc_adapt":
            if "ema_model_state_dict" in checkpoint:
                checkpoint = list(checkpoint["ema_model_state_dict"].values())[0]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.get_trainable_modules_list().load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dit":
            self.x_embedder.load_state_dict(get_submodule_weights(checkpoint, "x_embedder."), strict=True)
        elif self.cfg.pretrained_source == "dc-ae":
            if "ema_model_state_dict" in checkpoint:
                checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            target_state_dict = self.get_trainable_modules_list().state_dict()
            for key in target_state_dict:
                if key in checkpoint:
                    target_state_dict[key] = checkpoint[key]
                else:
                    target_state_dict[key] = checkpoint[key[2:]]
            self.get_trainable_modules_list().load_state_dict(target_state_dict)
        else:
            raise ValueError(f"Unsupported pretrained source {self.cfg.pretrained_source}")

    def initialize_weights(self) -> None:
        # Initialize patch_ffn
        if self.patch_ffn is not None:
            init_modules(self.patch_ffn, init_type="trunc_normal@0.02")
            # zero out the output of the patch_ffn
            for block in self.patch_ffn:
                assert isinstance(block, ResidualBlock) and isinstance(block.main, GLUMBConv)
                nn.init.constant_(block.main.point_conv.weight, 0)
                nn.init.constant_(block.main.point_conv.bias, 0)
                block.main.point_conv.weight.initialized = True
                block.main.point_conv.bias.initialized = True

    def check_initialization(self):
        for name, param in self.named_parameters():
            if not hasattr(param, "initialized"):
                raise ValueError(f"param {name} is not initialized")
            else:
                delattr(param, "initialized")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
