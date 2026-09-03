# Qwen-Image-Edit was introduced by Qwen Team.
# The original implementation is by Alibaba Cloud, licensed under the Apache License 2.0. See https://github.com/QwenLM/Qwen-Image.

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.utils.constants import DIFFUSERS_ATTN_BACKEND, DIFFUSERS_ATTN_CHECKS
from tqdm import tqdm

from ...models.utils.network import get_device
from ...t2icore.models.flux import (
    FeedForward,
    calculate_shift,
    calculate_shift_flux2,
)
from ...t2icore.models.flux_blocks.attention import FluxAttention
from .base_diffusion import BaseImageEditDiffusionModel, BaseImageEditDiffusionModelConfig
from .qwen_image_blocks.rotary_embed import apply_rotary_emb_qwen

__all__ = ["QwenImageConfig", "QwenImage"]


@dataclass
class QwenImageConfig(BaseImageEditDiffusionModelConfig):
    name: str = "QwenImage"
    eval_scheduler: str = "FluxScheduler"
    train_scheduler: str = "FlowMatchEulerDiscreteScheduler"
    num_inference_steps: int = 50

    patch_size: int = 2
    hidden_dim: int = 3072
    in_channels: int = 16

    # pos embedder
    rope_axes_dims: tuple[int] = (16, 56, 56)

    # caption embedder
    clip_dim: int = 768
    caption_dim: int = 4096
    text_max_length: int = 512
    text_encoder_id: str = "qwen-image-edit/qwen2.5-vl-bf16"

    # QwenImageBlocks
    depth: int = 60

    num_heads: int = 24
    joint_attention_dim: int = 3584
    attention_head_dim: int = 128

    freeze_backbone: bool = False
    head_only: bool = False
    load_type: str = "model_state_dict"

    drop_text_ratio: float = 0.0
    training_shift: Optional[float] = None

    # Flux2 shift: r = sqrt(latent_seq_len / base_seq_len)
    # set latent_seq_len = C * H * W of the training latent to align eval with training
    base_seq_len: int = 4096
    latent_seq_len: int = 8192


class QwenDoubleStreamAttnProcessor2_0:
    def native_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=None,
        )
        out = out.permute(0, 2, 1, 3)

        return out

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = self.native_attention(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            attn_mask=attention_mask,
            is_causal=False,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = FluxAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = hidden_states.dtype

        # Get modulation parameters for both streams
        img_mod_params = self.img_mod(temb)  # [B, 6*dim]
        txt_mod_params = self.txt_mod(temb)  # [B, 6*dim]

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states).to(dtype)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states).to(dtype)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=img_modulated,  # Image stream (will be processed as "sample")
            encoder_hidden_states=txt_modulated,  # Text stream (will be processed as "context")
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # QwenAttnProcessor2_0 returns (img_output, txt_output) when encoder_hidden_states is provided
        img_attn_output, txt_attn_output = attn_output

        # Apply attention gates and add residual (like in Megatron)
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states).to(dtype)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states).to(dtype)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # Clip to prevent overflow for fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImage(BaseImageEditDiffusionModel):
    def __init__(self, cfg: QwenImageConfig):
        super().__init__(cfg)
        self.cfg: QwenImageConfig

        if cfg.freeze_backbone or cfg.head_only:
            # freeze all parameters
            for parameter in self.parameters():
                parameter.requires_grad = False
            # unfreeze the parameters of selected modules
            if cfg.head_only:
                unfreeze_modules = [self.proj_out]
            else:
                unfreeze_modules = [self.img_in, self.proj_out]

            for m in unfreeze_modules:
                if m is not None:
                    for parameter in m.parameters():
                        parameter.requires_grad = True

    def build_model(self):
        from ...t2icore.models.flux_blocks.norm import AdaLayerNormContinuous, RMSNorm
        from .qwen_image_blocks.pos_embed import QwenEmbedRope
        from .qwen_image_blocks.time_embed import QwenTimestepProjEmbeddings

        self.out_channels = self.cfg.in_channels
        self.inner_dim = self.cfg.num_heads * self.cfg.attention_head_dim

        self.pos_embed = QwenEmbedRope(
            theta=10000, axes_dim=list(self.cfg.rope_axes_dims), max_image_seq_len=self.cfg.base_seq_len
        )

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(self.cfg.joint_attention_dim, eps=1e-6)

        self.img_in = nn.Linear(self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size, self.inner_dim)
        self.txt_in = nn.Linear(self.cfg.joint_attention_dim, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.cfg.num_heads,
                    attention_head_dim=self.cfg.attention_head_dim,
                )
                for _ in tqdm(range(self.cfg.depth))
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, self.cfg.patch_size * self.cfg.patch_size * self.out_channels, bias=True
        )

    def get_lora_target_modules(self) -> set[str]:
        target_modules_name_list = [
            "to_q",
            "to_k",
            "to_v",
            "add_k_proj",
            "add_v_proj",
            "add_q_proj",
            "to_out.0",
            "to_add_out",
            "img_mlp.net.0.proj",
            "img_mlp.net.2",
        ]
        target_modules = set()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                for target in target_modules_name_list:
                    if target in name:
                        target_modules.add(name)
                        break
        return target_modules

    def build_lora(self):
        super().build_lora()
        for name, params in self.img_in.named_parameters():
            params.requires_grad = True

        for name, params in self.proj_out.named_parameters():
            params.requires_grad = True

    def get_trainable_modules_list(self) -> nn.ModuleList:
        trainable_modules_list = []

        diffusion_model = {}
        for name, module in self.named_children():
            if name in [
                "pos_embed",
                "time_text_embed",
                "txt_norm",
                "img_in",
                "txt_in",
                "transformer_blocks",
                "norm_out",
                "proj_out",
            ]:
                diffusion_model[name] = module
            else:
                raise ValueError(f"module {name} is not supported")
        diffusion_model = nn.ModuleDict(diffusion_model)

        trainable_modules_list.append(diffusion_model)
        return nn.ModuleList(trainable_modules_list)

    def initialize_weights(self):
        #  Not Defined in FLUX codebase, transferred from SANA Codebase
        def _basic_init(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                torch.nn.init.xavier_uniform_(module.weight)
                module.weight.initialized = True
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    module.bias.initialized = True

        self.apply(_basic_init)

        nn.init.ones_(self.txt_norm.weight)
        self.txt_norm.weight.initialized = True

        #  As we use Attention from diffusers, it will be automatically initialized.
        for block in self.transformer_blocks:
            for module in [block.attn.norm_q, block.attn.norm_k, block.attn.norm_added_q, block.attn.norm_added_k]:
                module.weight.initialized = True

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "qwen_image":
            self.load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-gen":
            self.load_state_dict(checkpoint[self.cfg.load_type])
        else:
            raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
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

    def unpack_latents(self, latents, height, width):
        batch_size, _, channels = latents.shape

        latents = latents.view(
            batch_size,
            height // self.cfg.patch_size,
            width // self.cfg.patch_size,
            channels // (self.cfg.patch_size * self.cfg.patch_size),
            self.cfg.patch_size,
            self.cfg.patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (self.cfg.patch_size * self.cfg.patch_size), height, width)

        return latents

    def prepare_latents(self, latents: torch.Tensor, image_latents: torch.Tensor):
        batch_size, num_channels, h, w = image_latents.shape
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

    def enable_activation_checkpointing(self, mode: str):
        # checkpoint will not enable because FSDP optimizer resume
        if mode == "transformer":
            self.gradient_checkpointing = True
        else:
            self.gradient_checkpointing = False

    def forward_without_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        img_shapes: Optional[torch.Tensor] = None,
    ):
        text_embeddings = text_embed_info[self.cfg.text_encoder_id]["text_embeddings"]
        text_embedding_masks = text_embed_info[self.cfg.text_encoder_id]["text_embedding_masks"]
        txt_seq_lens = text_embedding_masks.sum(dim=1).tolist()

        x = self.img_in(x)

        timestep = timestep.to(x.dtype)
        text_embeddings = self.txt_norm(text_embeddings)
        text_embeddings = self.txt_in(text_embeddings)

        temb = self.time_text_embed(timestep, x)
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=x.device)

        for block in self.transformer_blocks:

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                text_embeddings, x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    text_embeddings,
                    text_embedding_masks,
                    temb,
                    image_rotary_emb,
                )

            else:
                text_embeddings, x = block(
                    x,
                    text_embeddings,
                    text_embedding_masks,
                    temb,
                    image_rotary_emb,
                )

        x = self.norm_out(x, temb, convert_norm_dtype=True)
        output = self.proj_out(x)

        return output

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        neg_text_embed_info: dict[str, dict[str, torch.Tensor]],
        image_embed_info: dict[str, Any],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        device = get_device(self)
        image_latents = image_embed_info["ae_feature"]
        bs, _, h, w = image_latents.shape
        if noise is None:
            noise = torch.randn(bs, self.cfg.in_channels, h, w, device=device, generator=generator)
            noise = noise.to(dtype=image_latents.dtype)

        img_shapes = [
            [
                (1, h // self.cfg.patch_size, w // self.cfg.patch_size),
            ]
            * 2
        ] * bs

        latents = noise
        latents, image_latents = self.prepare_latents(latents, image_latents)

        if self.cfg.eval_scheduler == "FluxScheduler":
            timesteps = self.eval_scheduler.get_timesteps(device=device)
            for t in tqdm(timesteps):
                if latents.shape[0] != image_latents.shape[0]:
                    latents = latents.unsqueeze(0)
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.forward_without_cfg(
                    x=latent_model_input,
                    timestep=timestep / 1000,
                    text_embed_info=text_embed_info,
                    img_shapes=img_shapes,
                )
                noise_pred = noise_pred[:, : latents.size(1)]

                neg_noise_pred = self.forward_without_cfg(
                    x=latent_model_input,
                    timestep=timestep / 1000,
                    text_embed_info=neg_text_embed_info,
                    img_shapes=img_shapes,
                )
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                comb_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit. No naming error.
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True).to(noise_pred.dtype)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).to(noise_pred.dtype)

                noise_pred = comb_pred * (cond_norm / noise_norm)

                latents = self.eval_scheduler.step(noise_pred, t, latents, return_dict=False)
        elif self.cfg.eval_scheduler == "Flux2Scheduler":
            timesteps = self.eval_scheduler.get_timesteps(device=device, image_seq_len=self.cfg.input_size**2)
            for t in tqdm(timesteps):
                if latents.shape[0] != image_latents.shape[0]:
                    latents = latents.unsqueeze(0)
                latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.forward_without_cfg(
                    x=latent_model_input,
                    timestep=timestep / 1000,
                    text_embed_info=text_embed_info,
                    img_shapes=img_shapes,
                )
                noise_pred = noise_pred[:, : latents.size(1)]

                neg_noise_pred = self.forward_without_cfg(
                    x=latent_model_input,
                    timestep=timestep / 1000,
                    text_embed_info=neg_text_embed_info,
                    img_shapes=img_shapes,
                )
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                comb_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit. No naming error.
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True).to(noise_pred.dtype)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).to(noise_pred.dtype)

                noise_pred = comb_pred * (cond_norm / noise_norm)

                latents = self.eval_scheduler.step(noise_pred, t, latents)
        else:
            raise NotImplementedError(f"eval_scheduler {self.cfg.eval_scheduler} is not supported")

        latents = self.unpack_latents(latents, h, w)

        return latents

    def forward_train(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        image_embed_info: dict[str, Any],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        info = {}
        detailed_loss_dict = {}

        from diffusers.training_utils import compute_density_for_timestep_sampling

        # Sample noise that we'll add to the latents
        latents = x
        image_latents = image_embed_info["ae_features"]
        bs, C, h, w = latents.shape
        latents, image_latents = self.prepare_latents(latents, image_latents)

        noise = torch.randn_like(latents).to(latents.device)

        img_shapes = [
            [
                (1, h // self.cfg.patch_size, w // self.cfg.patch_size),
            ]
            * 2
        ] * bs

        # Stage 1: Compute noisy_latents and timesteps (scheduler-specific)
        if self.cfg.train_scheduler == "FlowMatchEulerDiscreteScheduler":
            image_seq_len = h * w
            self.train_scheduler.set_timesteps(self.cfg.train_sampling_steps, mu=calculate_shift(image_seq_len))

            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bs,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=None,  # not used
            )
            indices = (u * self.train_scheduler.config.num_train_timesteps).long()
            timesteps = self.train_scheduler.timesteps[indices].to(device=latents.device)

            noisy_latents = self.train_scheduler.scale_noise(latents, timesteps, noise)
        elif self.cfg.train_scheduler == "FlowMatchEulerDiscreteSchedulerFlux2":
            u = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=bs,
                logit_mean=0.0,
                logit_std=1.0,
            )

            image_seq_len = h * w * C

            indices = (u * self.cfg.train_sampling_steps).long()
            sigmas = (self.cfg.train_sampling_steps - indices) / self.cfg.train_sampling_steps
            # Flux2 shift: t' = (r * t) / (1 + (r - 1) * t), where r = sqrt(image_seq_len / base_seq_len)
            mu = calculate_shift_flux2(image_seq_len, self.cfg.base_seq_len)
            sigmas = (mu * sigmas) / (1.0 + (mu - 1.0) * sigmas)
            sigmas = sigmas.to(device=latents.device)
            timesteps = sigmas * self.cfg.train_sampling_steps
            sigmas = sigmas.view(-1, 1, 1)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        latent_model_inputs = torch.cat([noisy_latents, image_latents], dim=1)
        noise_pred = self.forward_without_cfg(
            latent_model_inputs,
            timesteps / 1000,
            text_embed_info=text_embed_info,
            img_shapes=img_shapes,
        )[:, : latents.size(1)]

        target = noise - latents

        loss = torch.mean(((noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
        loss = loss.mean()

        detailed_loss_dict["loss"] = loss.item()
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
