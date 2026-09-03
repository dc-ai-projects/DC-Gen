# Modified from ``https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from ...models.utils.network import get_device
from .base_diffusion import BaseVideoGenDiffusionModel, BaseVideoGenDiffusionModelConfig

__all__ = ["WanT2VConfig", "WanT2V"]


@dataclass
class WanT2VConfig(BaseVideoGenDiffusionModelConfig):
    name: str = "WanT2V"
    eval_scheduler: str = "WanScheduler"
    train_scheduler: str = "FlowMatchScheduler"
    flow_shift: float = 5.0
    num_inference_steps: int = 50

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

    # DiTBlocks
    ffn_dim: int = 8960
    num_heads: int = 12
    window_size: tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_norm: bool = True
    norm_eps: float = 1e-6

    # Freeze Submodules
    freeze_backbone: bool = False
    freeze_text_embed: bool = False
    freeze_cross_attn: bool = False
    full_tune_patch_head: bool = True
    freeze_time_proj: bool = False
    only_apply_lora_to_backbone: bool = False


class MLPProj(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        return self.proj(image_embeds)


class WanHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        import math

        from .wan_blocks.norm import WanLayerNorm

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, expand_t=False):
        x_type = x.dtype
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if expand_t:
                e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
                x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
            else:
                e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
                x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x.to(x_type)


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        task_type="t2v",
        text_max_length=512,
        use_clip_feat=True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        from .wan_blocks.attention import WanI2VCrossAttention, WanSelfAttention, WanT2VCrossAttention
        from .wan_blocks.norm import WanLayerNorm

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)

        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        if task_type == "t2v" or not use_clip_feat:
            self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        elif task_type == "i2v":
            self.cross_attn = WanI2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps, text_max_length)
        else:
            raise NotImplementedError(f"Task can only be T2V or I2V")

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        expand_t=False,
    ):
        assert e.dtype == torch.float32
        with torch.amp.autocast("cuda", dtype=torch.float32):
            if expand_t:
                e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            else:
                e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        if expand_t:
            e = tuple(torch.squeeze(tensor, dim=2) for tensor in e)

        x_dtype = x.dtype

        y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).to(x_dtype), seq_lens, grid_sizes, freqs)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(x_dtype))
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, e)
        return x


class WanT2V(BaseVideoGenDiffusionModel):
    def __init__(self, cfg: WanT2VConfig):
        super().__init__(cfg)
        self.cfg: WanT2VConfig

        if self.cfg.freeze_backbone:
            # freeze all parameters
            for parameter in self.parameters():
                parameter.requires_grad = False
            # unfreeze the parameters of selected modules
            unfreeze_modules = [self.patch_embedding, self.head]
            for m in unfreeze_modules:
                if m is not None:
                    for parameter in m.parameters():
                        parameter.requires_grad = True
        if self.cfg.freeze_cross_attn:
            for block in self.blocks:
                for parameter in block.cross_attn.parameters():
                    parameter.requires_grad = False
        if self.cfg.freeze_text_embed:
            for parameter in self.text_embedding.parameters():
                parameter.requires_grad = False
            if hasattr(self, "img_emb"):
                for parameter in self.img_emb.parameters():
                    parameter.requires_grad = False
        if self.cfg.freeze_time_proj:
            for parameter in self.time_embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.time_projection.parameters():
                parameter.requires_grad = False
        self.unfrozen_params = [
            name.replace(".weight", "").replace(".bias", "")
            for name, param in self.named_parameters()
            if param.requires_grad
        ]

        self.gradient_checkpointing = False

    def build_backbone(self, task_type: str, use_clip_feat: bool):
        self.patch_size = self.cfg.patch_size

        self.text_embedding = nn.Sequential(
            nn.Linear(self.cfg.caption_channels, self.cfg.hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(self.cfg.freq_dim, self.cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.cfg.hidden_size, 6 * self.cfg.hidden_size))

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

        self.blocks = nn.ModuleList(
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

        self.out_channels = self.cfg.in_channels
        self.head = WanHead(
            self.cfg.hidden_size,
            self.out_channels,
            self.patch_size,
            self.cfg.norm_eps,
        )

    def build_model(self):
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

        self.build_backbone("t2v", False)

    def lora_wrap(self, lora_rank, lora_alpha, use_lora, use_dora):
        from peft import LoraConfig, get_peft_model, inject_adapter_in_model

        no_grad_params = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                no_grad_params.add(name.replace(".weight", "").replace(".bias", ""))
        all_linear_layers = []
        if self.cfg.only_apply_lora_to_backbone:
            for name, module in self.blocks.named_modules():
                if isinstance(module, nn.Linear):
                    all_linear_layers.append("blocks." + name)
        else:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear) and (not self.cfg.full_tune_patch_head or "head" not in name):
                    all_linear_layers.append(name)
        lora_modules = list(set(all_linear_layers) - set(no_grad_params))

        if use_lora and use_dora:  # Mixture
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.0,
                bias="none",
                use_dora=False,
            )
            dora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.0,
                bias="none",
                use_dora=True,
            )

            self = get_peft_model(self, lora_config)
            self.add_adapter(adapter_name="dora_adapter", peft_config=dora_config)

        else:
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_modules,
                lora_dropout=0.0,
                bias="none",
                use_dora=use_dora,
            )
            self = inject_adapter_in_model(config, self)

        for name, param in self.named_parameters():
            if "lora" not in name:
                if self.cfg.only_apply_lora_to_backbone and "blocks" not in name:
                    param.requires_grad = True
                    continue
                if self.cfg.full_tune_patch_head and ("patch_embedding" in name or "head" in name):
                    param.requires_grad = True
                    continue
                param.requires_grad = False
            else:
                base_name = name.split(".lora_")[0]
                if base_name in no_grad_params:
                    param.requires_grad = False

    def initialize_weights(self):
        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.initialized = True
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    m.bias.initialized = True

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        fan_in = self.cfg.in_channels * math.prod(self.cfg.patch_size)
        bound = 1 / (fan_in**0.5)
        nn.init.uniform_(self.patch_embedding.bias, -bound, bound)
        self.patch_embedding.weight.initialized = True
        self.patch_embedding.bias.initialized = True

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                m.weight.initialized = True
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                m.weight.initialized = True

        # init zero for v_img in each block
        for block in self.blocks:
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
        nn.init.zeros_(self.head.head.weight)
        self.head.head.weight.initialized = True
        nn.init.normal_(self.head.modulation, mean=0.0, std=1.0 / (self.cfg.hidden_size**0.5))
        self.head.modulation.initialized = True  # Constant Random Parameter

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
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
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=False)
        if self.cfg.pretrained_source == "wan":
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            self.load_state_dict(checkpoint)
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

            self.get_trainable_modules_list().load_state_dict(new_ckpt)
        elif self.cfg.pretrained_source == "dc-ae-fsdp":
            if "ema_model_state_dict" in checkpoint:
                checkpoint = next(iter(checkpoint["ema_model_state_dict"].values()))
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    def enable_activation_checkpointing(self, mode: str):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        if mode == "transformer":
            self.gradient_checkpointing = True
        elif mode == "full":
            self.patch_embedding = checkpoint_wrapper(self.patch_embedding, preserve_rng_state=False)
            self.time_embedding = checkpoint_wrapper(self.time_embedding, preserve_rng_state=False)
            self.time_projection = checkpoint_wrapper(self.time_projection, preserve_rng_state=False)
            self.text_embedding = checkpoint_wrapper(self.text_embedding, preserve_rng_state=False)
            for i in range(len(self.blocks)):
                self.blocks[i] = checkpoint_wrapper(self.blocks[i], preserve_rng_state=False)
            self.head = checkpoint_wrapper(self.head, preserve_rng_state=False)
            self.gradient_checkpointing = True
        else:
            raise ValueError(f"mode {mode} is not supported")

    def unpatchify(self, x, grid_sizes):
        bs, c, v = x.shape[0], self.out_channels, grid_sizes[0]
        x = x[:, : math.prod(v)].view(bs, *v, *self.patch_size, c)
        x = torch.einsum("bfhwpqrc->bcfphqwr", x)
        x = x.reshape(x.shape[0], c, *[v_ * p_ for v_, p_ in zip(v, self.patch_size)])
        return x

    def forward_without_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype

        bs = x.shape[0]
        x = x.to(dtype)
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
                t = t.flatten().repeat(seq_lens[0])
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.cfg.freq_dim, t).unflatten(0, (bt, seq_lens[0])).float()
                )
                e0 = self.time_projection(e).unflatten(2, (6, self.cfg.hidden_size))
            else:
                e = self.time_embedding(sinusoidal_embedding_1d(self.cfg.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.cfg.hidden_size))

        context = self.text_embedding(context).to(dtype)

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

        device = get_device(self)
        bs = text_embeddings.shape[0]
        if "null_text_embeddings" in text_embed_info[self.cfg.text_encoder_id]:
            null_text_embeddings = text_embed_info[self.cfg.text_encoder_id]["null_text_embeddings"]
        else:
            null_text_embeddings = self.null_embedding.unsqueeze(0).repeat(bs, 1, 1)

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
            for t in timesteps:
                samples = self.eval_scheduler.step(
                    model=self.forward_without_cfg,
                    latents=samples,
                    timestep=t,
                    text_embeddings=text_embeddings,
                    null_text_embeddings=null_text_embeddings,
                    cfg_scale=cfg_scale,
                    **image_embed_info,
                )
        else:
            raise ValueError(f"Eval scheduler {self.cfg.eval_scheduler} is not supported.")

        return samples, {}

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
            scheduler_output = self.training_scheduler.training_losses(
                self.forward_without_cfg, x, timesteps, model_kwargs=dict(y=y)
            )
            loss = scheduler_output["loss"].mean()
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        detailed_loss_dict["loss"] = loss
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
