# Modified from ``https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import os
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from ...models.utils.network import get_device
from .wan_t2v import MLPProj, WanT2V, WanT2VConfig

__all__ = ["WanI2VConfig", "WanI2V"]


@dataclass
class WanI2VConfig(WanT2VConfig):
    name: str = "WanI2V"
    text_encoder_id: str = "wan2.1-i2v/umt5-512-bf16"

    flow_shift: float = 5.0
    use_mask: bool = True

    i2v_concat: bool = True
    use_clip_feat: bool = True

    t_ratio: int = 4


class WanI2V(WanT2V):
    def __init__(self, cfg: WanI2VConfig):
        super().__init__(cfg)
        self.cfg: WanI2VConfig

    def build_model(self):
        if self.cfg.i2v_concat:
            self.patch_embedding = nn.Conv3d(
                self.cfg.in_channels * 2 + self.cfg.t_ratio,
                self.cfg.hidden_size,
                kernel_size=self.cfg.patch_size,
                stride=self.cfg.patch_size,
            )
        else:
            self.patch_embedding = nn.Conv3d(
                self.cfg.in_channels,
                self.cfg.hidden_size,
                kernel_size=self.cfg.patch_size,
                stride=self.cfg.patch_size,
            )

        null_embedding_path = os.path.join(
            "assets/data/null_text_embeddings",
            f"{self.cfg.text_encoder_id}.pth",
        )
        null_embedding = torch.load(null_embedding_path, weights_only=True, map_location="cpu")
        self.register_buffer("null_embedding", null_embedding)

        if self.cfg.use_clip_feat:
            self.img_emb = MLPProj(1280, self.cfg.hidden_size)

        self.build_backbone("i2v", self.cfg.use_clip_feat)

    def initialize_weights(self):
        super().initialize_weights()

        if not self.cfg.use_clip_feat:
            return

        for layer_id in [0, 4]:
            nn.init.ones_(self.img_emb.proj[layer_id].weight)
            self.img_emb.proj[layer_id].weight.initialized = True
            nn.init.zeros_(self.img_emb.proj[layer_id].bias)
            self.img_emb.proj[layer_id].bias.initialized = True

        for block in self.blocks:
            nn.init.ones_(block.cross_attn.norm_k_img.weight)
            block.cross_attn.norm_k_img.weight.initialized = True

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
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

    def forward_without_cfg(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, ae_feature=None, vlm_feature=None, **kwargs
    ) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype

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

        x = self.patch_embedding(x)

        grid_sizes = torch.tensor(x.shape[2:]).unsqueeze(0).repeat(bs, 1).to(torch.long)
        x = x.flatten(2).transpose(1, 2).contiguous()
        seq_lens = torch.tensor(x.size(1)).to(torch.long).repeat(bs)

        from .wan_blocks.pos_embed import sinusoidal_embedding_1d

        with torch.amp.autocast("cuda", dtype=torch.float32):
            if self.cfg.expand_t:
                bt = t.size(0)
                t = t[:, None].repeat((1, seq_lens[0]))
                assert seq_lens[0] % x_len == 0
                assert all(seq_len == seq_lens[0] for seq_len in seq_lens)
                t[:, : seq_lens[0] // x_len] = 0
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.cfg.freq_dim, t).unflatten(0, (bt, seq_lens[0])).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.cfg.hidden_size))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.cfg.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.cfg.hidden_size))

        context = self.text_embedding(context).to(dtype)
        if vlm_feature is not None:
            context_clip = self.img_emb(vlm_feature)
            context = torch.concat([context_clip, context], dim=1)

        for block in self.blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    e0,
                    seq_lens,
                    grid_sizes,
                    self.freqs,
                    context,
                    self.cfg.expand_t,
                    **ckpt_kwargs,
                )
            else:
                x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, self.cfg.expand_t)

        x = self.head(x, e, expand_t=self.cfg.expand_t)

        x = self.unpatchify(x, grid_sizes)

        return x.to(torch.float32)

    def forward_train(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        image_embed_info: dict[str, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], dict]:
        info = {}
        detailed_loss_dict = {}
        device = x.device
        y = text_embed_info[self.cfg.text_encoder_id]["text_embeddings"]

        if self.cfg.train_scheduler == "FlowMatchScheduler":
            bs = x.shape[0]
            timesteps = torch.randint(0, self.cfg.train_sampling_steps, (bs,), device=device)
            if self.cfg.i2v_concat:
                ae_feature = torch.cat((image_embed_info["img_masks"], image_embed_info["ae_feature"]), dim=1)
            else:
                ae_feature = image_embed_info["ae_feature"]
            if self.cfg.use_clip_feat:
                vlm_feature = image_embed_info["vlm_feature"]
            else:
                vlm_feature = None
            scheduler_output = self.training_scheduler.training_losses(
                self.forward_without_cfg,
                x,
                timesteps,
                model_kwargs=dict(y=y, ae_feature=ae_feature, vlm_feature=vlm_feature),
            )
            loss = scheduler_output["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        detailed_loss_dict["loss"] = loss
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
