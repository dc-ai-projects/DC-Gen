# Modified from ``https://https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/model.py''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ...models.utils.network import get_device
from .base_diffusion import BaseVideoGenDiffusionModel, BaseVideoGenDiffusionModelConfig
from .wan_t2v import WanAttentionBlock, WanHead

__all__ = ["MoEWanT2VConfig", "MoEWanT2V"]


@dataclass
class MoEWanT2VConfig(BaseVideoGenDiffusionModelConfig):
    name: str = "MoEWanT2V"
    eval_scheduler: str = "WanScheduler"
    train_scheduler: str = "FlowMatchScheduler"
    flow_shift: float = 12.0
    num_inference_steps: int = 40

    patch_size: tuple[int, int, int] = (1, 2, 2)
    input_size: tuple[int, int, int] = (21, 60, 104)
    hidden_size: int = 1536
    depth: int = 30
    pos_embed_type: str = "sincos"

    # time embedder
    freq_dim: int = 256
    expand_t: bool = False  # For Wan2.2

    # caption embedder
    caption_channels: int = 4096
    class_dropout_prob: float = 0.1
    text_max_length: int = 512
    y_norm_scale_factor: float = 0.01
    text_encoder_id: str = "wan2.1-t2v/umt5-512-bf16"

    # SanaBlocks
    ffn_dim: int = 8960
    num_heads: int = 12
    window_size: tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_norm: bool = True
    norm_eps: float = 1e-6

    # MoE
    boundaries: tuple[float, ...] = (0.875,)
    cfg_scales: tuple[float, ...] = (4.0, 3.0)

    # Offload
    offload: bool = True


class MoEWanT2V(BaseVideoGenDiffusionModel):
    def __init__(self, cfg: MoEWanT2VConfig):
        self.num_submodels = len(cfg.pretrained_paths)
        assert self.num_submodels >= 2

        super().__init__(cfg)
        self.cfg: MoEWanT2VConfig

        assert len(self.cfg.boundaries) == self.num_submodels - 1
        assert len(self.cfg.cfg_scales) == self.num_submodels
        self.boundaries = [boundary * 1000 for boundary in self.cfg.boundaries]

    def build_backbone(self, task_type: str, use_clip_feat: bool, idx: int):
        self.submodel[idx].text_embedding = nn.Sequential(
            nn.Linear(self.cfg.caption_channels, self.cfg.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
        )

        self.submodel[idx].time_embedding = nn.Sequential(
            nn.Linear(self.cfg.freq_dim, self.cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
        )
        self.submodel[idx].time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(self.cfg.hidden_size, 6 * self.cfg.hidden_size)
        )

        self.submodel[idx].blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    self.cfg.hidden_size,
                    self.cfg.ffn_dim,
                    self.cfg.num_heads,
                    self.cfg.window_size,
                    self.cfg.qk_norm,
                    self.cfg.cross_norm,
                    self.cfg.norm_eps,
                    task_type,
                    self.cfg.text_max_length,
                    use_clip_feat,
                )
                for _ in tqdm(range(self.cfg.depth), desc=f"Building {self.cfg.depth} transformer blocks")
            ]
        )

        self.submodel[idx].head = WanHead(
            self.cfg.hidden_size,
            self.out_channels,
            self.patch_size,
            self.cfg.norm_eps,
        )

    def build_model(self):
        self.submodel = nn.ModuleList([nn.Module() for _ in range(self.num_submodels)])
        self.patch_size = self.cfg.patch_size
        for idx in range(self.num_submodels):
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
            self.build_backbone("t2v", False, idx)

    def initialize_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.initialized = True
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    m.bias.initialized = True

        for idx in range(self.num_submodels):
            # init embeddings
            nn.init.xavier_uniform_(self.submodel[idx].patch_embedding.weight.flatten(1))
            fan_in = self.cfg.in_channels * math.prod(self.cfg.patch_size)
            bound = 1 / (fan_in**0.5)
            nn.init.uniform_(self.submodel[idx].patch_embedding.bias, -bound, bound)
            self.submodel[idx].patch_embedding.weight.initialized = True
            self.submodel[idx].patch_embedding.bias.initialized = True

            for m in self.submodel[idx].text_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    m.weight.initialized = True
            for m in self.submodel[idx].time_embedding.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                    m.weight.initialized = True

            # init zero for v_img in each block
            for block in self.submodel[idx].blocks:
                nn.init.normal_(block.modulation, mean=0.0, std=1.0 / (self.cfg.hidden_size**0.5))
                block.modulation.initialized = True

                nn.init.ones_(block.self_attn.norm_q.weight)
                block.self_attn.norm_q.weight.initialized = True
                nn.init.ones_(block.self_attn.norm_k.weight)
                block.self_attn.norm_k.weight.initialized = True
                nn.init.ones_(block.cross_attn.norm_q.weight)
                block.cross_attn.norm_q.weight.initialized = True
                nn.init.ones_(block.cross_attn.norm_k.weight)
                block.cross_attn.norm_k.weight.initialized = True
                nn.init.ones_(block.norm3.weight)
                block.norm3.weight.initialized = True
                nn.init.zeros_(block.norm3.bias)
                block.norm3.bias.initialized = True

            # init output layer
            nn.init.zeros_(self.submodel[idx].head.head.weight)
            self.submodel[idx].head.head.weight.initialized = True
            nn.init.normal_(self.submodel[idx].head.modulation, mean=0.0, std=1.0 / (self.cfg.hidden_size**0.5))
            self.submodel[idx].head.modulation.initialized = True  # Constant Random Parameter

    def get_trainable_modules_list(self, submodel) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in submodel.named_children():
            if name in [
                "patch_embedding",
                "text_embedding",
                "time_embedding",
                "time_projection",
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

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, idx: int) -> torch.Tensor:
        device = self.submodel[idx].patch_embedding.weight.device
        dtype = self.submodel[idx].patch_embedding.weight.dtype

        bs = x.shape[0]
        x = x.to(dtype)
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
                e = self.submodel[idx].time_embedding(
                    sinusoidal_embedding_1d(self.cfg.freq_dim, t).unflatten(0, (bt, seq_lens[0])).float()
                )
                e0 = self.submodel[idx].time_projection(e).unflatten(2, (6, self.cfg.hidden_size))
            else:
                e = self.submodel[idx].time_embedding(sinusoidal_embedding_1d(self.cfg.freq_dim, t).float())
                e0 = self.submodel[idx].time_projection(e).unflatten(1, (6, self.cfg.hidden_size))

        context = self.submodel[idx].text_embedding(context).to(dtype)

        for block in self.submodel[idx].blocks:
            x = block(x, e0, seq_lens, grid_sizes, self.freqs, context, self.cfg.expand_t)

        x = self.submodel[idx].head(x, e, expand_t=self.cfg.expand_t)

        x = self.unpatchify(x, grid_sizes)

        return x.to(torch.float32)

    def split_timesteps_into_segments(self, timesteps, boundaries):
        boundaries = sorted(boundaries, reverse=True)
        segments = []

        mask1 = timesteps >= boundaries[0]
        segments.append(timesteps[mask1])

        for i in range(len(boundaries) - 1):
            mask = (timesteps >= boundaries[i + 1]) & (timesteps < boundaries[i])
            segments.append(timesteps[mask])

        mask_last = timesteps < boundaries[-1]
        segments.append(timesteps[mask_last])

        return segments

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        image_embed_info: dict[str, Any],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 5.0,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        text_embeddings = text_embed_info[self.cfg.text_encoder_id]["text_embeddings"]
        device = get_device(self.submodel[0])
        bs = text_embeddings.shape[0]
        null_text_embeddings = self.submodel[0].null_embedding.unsqueeze(0).repeat(bs, 1, 1)

        if noise is None:
            if image_embed_info == {}:
                z = torch.randn(
                    bs,
                    self.cfg.in_channels,
                    self.cfg.input_size[0],
                    self.cfg.input_size[1],
                    self.cfg.input_size[2],
                    device=device,
                    generator=generator,
                )
            else:
                h, w = image_embed_info["ae_feature"].shape[-2], image_embed_info["ae_feature"].shape[-1]
                z = torch.randn(
                    bs,
                    self.cfg.in_channels,
                    self.cfg.input_size[0],
                    h,
                    w,
                    device=device,
                    generator=generator,
                )
        else:
            z = noise

        if self.cfg.eval_scheduler == "DPMS":
            from ...c2icore.diffusioncore.models.sana_utils.dpm_solver import DPMS

            dpm_solver = DPMS(
                self.forward_without_cfg,
                condition=text_embeddings,
                uncondition=null_text_embeddings,
                guidance_type=self.cfg.guidance_type,
                cfg_scale=cfg_scale,
                pag_scale=pag_scale,
                pag_applied_layers=self.cfg.pag_applied_layers,
                model_type="flow",
                schedule="FLOW",
                interval_guidance=self.cfg.interval_guidance,
            )
            samples = dpm_solver.sample(
                z,
                steps=self.cfg.num_inference_steps,
                order=2,
                skip_type="time_uniform_flow",
                method="multistep",
                flow_shift=self.cfg.flow_shift,
            )
        elif self.cfg.eval_scheduler == "WanScheduler":
            samples = z
            timesteps = self.eval_scheduler.set_timesteps(
                num_inference_steps=self.cfg.num_inference_steps,
                flow_shift=self.cfg.flow_shift,
                device=device,
            )
            segments = self.split_timesteps_into_segments(timesteps, self.boundaries)

            for idx, segment in enumerate(segments):
                for seg_t in segment:
                    samples = self.eval_scheduler.step(
                        model=self.forward_without_cfg,
                        latents=samples,
                        timestep=seg_t,
                        text_embeddings=text_embeddings,
                        null_text_embeddings=null_text_embeddings,
                        idx=idx,
                        cfg_scale=self.cfg.cfg_scales[idx],
                        **image_embed_info,
                    )
                if self.cfg.offload and idx < len(segments) - 1:
                    self.submodel[idx] = self.submodel[idx].to("cpu")
                    torch.cuda.empty_cache()
                    self.submodel[idx + 1] = self.submodel[idx + 1].to(device)
        else:
            raise ValueError(f"Eval scheduler {self.cfg.eval_scheduler} is not supported.")

        return samples, {}
