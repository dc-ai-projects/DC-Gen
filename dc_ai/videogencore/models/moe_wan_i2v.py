# Modified from ``https://https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import math
import os
from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn

from .moe_wan_t2v import MoEWanT2V, MoEWanT2VConfig
from .wan_t2v import MLPProj

__all__ = ["MoEWanI2VConfig", "MoEWanI2V"]


@dataclass
class MoEWanI2VConfig(MoEWanT2VConfig):
    name: str = "MoEWanI2V"
    text_encoder_id: str = "wan2.1-i2v/umt5-512-bf16"
    flow_shift: float = 5.0

    use_mask: bool = True

    i2v_concat: bool = True
    use_clip_feat: bool = True

    t_ratio: int = 4

    # MoE
    boundaries: tuple[float, ...] = (0.9,)
    cfg_scales: tuple[float, ...] = (3.5, 3.5)


class MoEWanI2V(MoEWanT2V):
    def __init__(self, cfg: MoEWanI2VConfig):
        super().__init__(cfg)
        self.cfg: MoEWanI2VConfig

    def build_model(self):
        self.submodel = nn.ModuleList([nn.Module() for _ in range(self.num_submodels)])
        self.patch_size = self.cfg.patch_size
        for idx in range(self.num_submodels):
            if self.cfg.i2v_concat:
                self.submodel[idx].patch_embedding = nn.Conv3d(
                    self.cfg.in_channels * 2 + self.cfg.t_ratio,
                    self.cfg.hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                )
            else:
                self.submodel[idx].patch_embedding = nn.Conv3d(
                    self.cfg.in_channels,
                    self.cfg.hidden_size,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                )

        null_embedding_path = os.path.join(
            "assets/data/null_text_embeddings",
            f"{self.cfg.text_encoder_id}.pth",
        )
        null_embedding = torch.load(null_embedding_path, weights_only=True)

        def rope_params(max_seq_len, dim, theta=10000):
            assert dim % 2 == 0
            freqs = torch.outer(
                torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
            )
            freqs = torch.polar(torch.ones_like(freqs), freqs)
            return freqs

        assert (self.cfg.hidden_size % self.cfg.num_heads) == 0 and (
            self.cfg.hidden_size // self.cfg.num_heads
        ) % 2 == 0
        d = self.cfg.hidden_size // self.cfg.num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )
        self.out_channels = self.cfg.in_channels

        for idx in range(self.num_submodels):
            # We must define null embedding for each submodel, in order to prevent weight loading error
            self.submodel[idx].register_buffer("null_embedding", null_embedding)
            if self.cfg.use_clip_feat:
                self.submodel[idx].img_emb = MLPProj(1280, self.cfg.hidden_size)
            self.build_backbone("i2v", False, idx)

    def initialize_weights(self):
        super().initialize_weights()

        if not self.cfg.use_clip_feat:
            return

        for idx in range(self.num_submodels):
            for layer_id in [0, 4]:
                nn.init.ones_(self.submodel[idx].img_emb.proj[layer_id].weight)
                self.submodel[idx].img_emb.proj[layer_id].weight.initialized = True
                nn.init.zeros_(self.submodel[idx].img_emb.proj[layer_id].bias)
                self.submodel[idx].img_emb.proj[layer_id].bias.initialized = True

            for block in self.submodel[idx].blocks:
                nn.init.ones_(block.cross_attn.norm_k_img.weight)
                block.cross_attn.norm_k_img.weight.initialized = True

    def get_trainable_modules_list(self, submodel) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in submodel.named_children():
            if name in [
                "patch_embedding",
                "text_embedding",
                "time_embedding",
                "time_projection",
                "img_emb",
                "blocks",
                "head",
            ]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)
        for name, parameter in self.named_parameters(recurse=False):
            raise ValueError(f"parameter {name} is not supported")

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def load_model(self):
        for idx in range(self.num_submodels):
            checkpoint = torch.load(self.cfg.pretrained_paths[idx], map_location="cpu", weights_only=False)
            if self.cfg.pretrained_source == "wan":
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                self.submodel[idx].load_state_dict(checkpoint)
            elif self.cfg.pretrained_source == "dc-ae":
                if "ema_model_state_dict" in checkpoint:
                    checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                new_ckpt = {}
                for key in checkpoint.keys():
                    if "pos_embed" in key or "null_embedding" in key:
                        continue
                    new_ckpt["0." + key] = checkpoint[key]

                self.get_trainable_modules_list(self.submodel[idx]).load_state_dict(new_ckpt)
            elif self.cfg.pretrained_source == "dc-ae-fsdp":
                if "ema_model_state_dict" in checkpoint:
                    checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
                self.submodel[idx].load_state_dict(checkpoint)
            else:
                raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    def unpatchify(self, x, grid_sizes):
        bs, c, v = x.shape[0], self.out_channels, grid_sizes[0]
        x = x[:, : math.prod(v)].view(bs, *v, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(x.shape[0], c, *[v_ * p_ for v_, p_ in zip(v, self.patch_size)])
        return x

    def forward_without_cfg(self, x, t, y, ae_feature=None, vlm_feature=None, idx=None):
        device = self.submodel[idx].patch_embedding.weight.device
        dtype = self.submodel[idx].patch_embedding.weight.dtype

        bs, x_len = x.shape[0], x.shape[2]
        if ae_feature is not None:
            if self.cfg.i2v_concat:
                x = torch.cat((x, ae_feature), dim=1).to(dtype)
            else:
                x[:, :, 0] = ae_feature[:, :, 0]

        context = y.to(dtype)
        self.freqs = self.freqs.to(device)

        if self.training:
            drop_ids = torch.rand(bs).cuda() < self.cfg.class_dropout_prob
            context = torch.where(drop_ids[:, None, None], self.null_embedding, context)

        x = self.submodel[idx].patch_embedding(x)

        grid_sizes = torch.tensor(x.shape[2:]).unsqueeze(0).repeat(bs, 1).to(torch.long)
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor(x.size(1)).to(torch.long).repeat(bs)

        from .wan_blocks.pos_embed import sinusoidal_embedding_1d

        with torch.amp.autocast("cuda", dtype=torch.float32):
            if self.cfg.expand_t:
                bt = t.size(0)
                t = t.flatten().repeat(seq_lens[0])
                if not self.cfg.i2v_concat:
                    assert seq_lens[0] % x_len == 0
                    t[: seq_lens[0] // x_len] = 0
                e = self.submodel[idx].time_embedding(
                    sinusoidal_embedding_1d(self.cfg.freq_dim, t).unflatten(0, (bt, seq_lens[0])).float()
                )
                e0 = self.submodel[idx].time_projection(e).unflatten(2, (6, self.cfg.hidden_size))
            else:
                e = self.submodel[idx].time_embedding(sinusoidal_embedding_1d(self.cfg.freq_dim, t).float())
                e0 = self.submodel[idx].time_projection(e).unflatten(1, (6, self.cfg.hidden_size))

        context = self.submodel[idx].text_embedding(context).to(dtype)
        if vlm_feature is not None:
            context_clip = self.submodel[idx].img_emb(vlm_feature)
            context = torch.concat([context_clip, context], dim=1)

        for block in self.submodel[idx].blocks:
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, self.cfg.expand_t)

        x = self.submodel[idx].head(x, e, expand_t=self.cfg.expand_t)

        x = self.unpatchify(x, grid_sizes)

        return x.to(torch.float32)
