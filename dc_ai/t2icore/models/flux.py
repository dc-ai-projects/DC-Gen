# FLUX was introduced by Black Forest Lab.
# The original implementation is by Black Forest Lab, licensed under the Apache License 2.0. See https://github.com/black-forest-labs/flux.

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ...models.utils.network import get_device
from .base_diffusion import BaseT2IDiffusionModel, BaseT2IDiffusionModelConfig
from .flux_blocks.attention import FluxAttention

__all__ = ["FluxConfig", "Flux"]


@dataclass
class FluxConfig(BaseT2IDiffusionModelConfig):
    name: str = "Flux"
    eval_scheduler: str = "FluxScheduler"
    train_scheduler: str = "SanaScheduler"
    num_inference_steps: int = 20

    patch_size: int = 2
    hidden_dim: int = 3072
    in_channels: int = 16

    # pos embedder
    rope_axes_dims: tuple[int] = (16, 56, 56)
    pos_embed_type: str = "sincos"

    # caption embedder
    clip_dim: int = 768
    caption_dim: int = 4096
    clip_text_encoder_id: str = "flux.1-dev/clip-77-bf16"
    t5_text_encoder_id: str = "flux.1-dev/t5-512-bf16"

    # FluxBlocks
    depth: int = 57
    double_depth: int = 19
    single_depth: int = 38

    num_heads: int = 24
    mlp_ratio: float = 4.0
    joint_attention_dim: int = 4096
    attention_head_dim: int = 128
    norm_scale_factor: float = 0.01

    freeze_backbone: bool = False
    head_only: bool = False
    null_embeds_dir: Optional[str] = None
    load_type: str = "model_state_dict"

    use_guide_loss: bool = False
    drop_text_raio: float = 0.0
    training_shift: Optional[float] = None

    # Flux2 shift: r = sqrt(latent_seq_len / base_seq_len)
    # set latent_seq_len = C * H * W of the training latent to align eval with training
    latent_seq_len: int = 4096
    base_seq_len: int = 4096


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        act_fn = GELU(dim, inner_dim, approximate="tanh", bias=True)

        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(dropout))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=True))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class FluxTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, eps=1e-6, processor_type="FluxAttnProcessor2_0"):
        super().__init__()

        from .flux_blocks.norm import AdaLayerNormZero

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        if processor_type == "FluxAttnProcessor":
            from .flux_blocks.attention import FluxAttnProcessor

            processor = FluxAttnProcessor()
        elif processor_type == "FluxAttnProcessor2_0":
            from .flux_blocks.attention import FluxAttnProcessor2_0

            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(f"{processor_type} is not supported")

        self.attn = FluxAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # Attention.
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        attn_output, context_attn_output = attention_outputs

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states).to(hidden_states.dtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states).to(hidden_states.dtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class FluxSingleTransformerBlock(nn.Module):
    def __init__(
        self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0, processor_type="FluxAttnProcessor2_0"
    ):
        super().__init__()

        from .flux_blocks.norm import AdaLayerNormZeroSingle

        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if processor_type == "FluxAttnProcessor":
            from .flux_blocks.attention import FluxAttnProcessor

            processor = FluxAttnProcessor()
        elif processor_type == "FluxAttnProcessor2_0":
            from .flux_blocks.attention import FluxAttnProcessor2_0

            processor = FluxAttnProcessor2_0()
        else:
            raise ValueError(f"{processor_type} is not supported")

        self.attn = FluxAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


def calculate_shift(
    image_seq_len: int = 4096,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calculate_shift_flux2(image_seq_len: int, base_seq_len: int = 4096) -> float:
    """shift ratio r = sqrt(image_seq_len / base_seq_len).

    sigma' = (r * sigma) / (1 + (r - 1) * sigma)
    """
    return math.sqrt(image_seq_len / base_seq_len)


class Flux(BaseT2IDiffusionModel):
    def __init__(self, cfg: FluxConfig):
        super().__init__(cfg)
        self.cfg: FluxConfig

        if cfg.freeze_backbone or cfg.head_only:
            # freeze all parameters
            for parameter in self.parameters():
                parameter.requires_grad = False
            # unfreeze the parameters of selected modules
            if cfg.head_only:
                unfreeze_modules = [self.proj_out]
            else:
                unfreeze_modules = [self.x_embedder, self.proj_out]

            for m in unfreeze_modules:
                if m is not None:
                    for parameter in m.parameters():
                        parameter.requires_grad = True

    def build_model(self):
        from .flux_blocks.norm import AdaLayerNormContinuous
        from .flux_blocks.pos_embed import FluxPosEmbed
        from .ops.combined_embed import FluxCombinedEmbed

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=self.cfg.rope_axes_dims)

        self.time_text_embed = FluxCombinedEmbed(
            embedding_dim=self.cfg.hidden_dim,
            pooled_projection_dim=self.cfg.clip_dim,
        )

        if self.cfg.null_embeds_dir:
            t5_null_embedding_path = os.path.join(self.cfg.null_embeds_dir, f"{self.cfg.t5_text_encoder_id}.pth")
            clip_null_embedding_path = os.path.join(self.cfg.null_embeds_dir, f"{self.cfg.clip_text_encoder_id}.pth")

            t5_null_embedding = torch.load(t5_null_embedding_path, weights_only=True, map_location="cpu")
            clip_null_embedding = torch.load(clip_null_embedding_path, weights_only=True, map_location="cpu")
            self.register_buffer("t5_null_embedding", t5_null_embedding, persistent=False)
            self.register_buffer("clip_null_embedding", clip_null_embedding, persistent=False)

        self.context_embedder = nn.Linear(self.cfg.joint_attention_dim, self.cfg.hidden_dim)
        self.x_embedder = nn.Linear(
            self.cfg.in_channels * self.cfg.patch_size * self.cfg.patch_size, self.cfg.hidden_dim
        )

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.cfg.hidden_dim,
                    num_attention_heads=self.cfg.num_heads,
                    attention_head_dim=self.cfg.attention_head_dim,
                )
                for i in range(self.cfg.double_depth)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.cfg.hidden_dim,
                    num_attention_heads=self.cfg.num_heads,
                    attention_head_dim=self.cfg.attention_head_dim,
                )
                for i in range(self.cfg.single_depth)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.proj_out = nn.Linear(
            self.cfg.hidden_dim, self.cfg.patch_size * self.cfg.patch_size * self.cfg.in_channels, bias=True
        )

    def get_lora_target_modules(self):
        target_modules_name_list = [
            "proj_mlp",
            "proj_out",
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        target_modules = set()
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.Linear):
                for target in target_modules_name_list:
                    if target in name:
                        target_modules.add(name)
        target_modules.remove("proj_out")
        return target_modules

    def build_lora(self):
        super().build_lora()
        for name, params in self.x_embedder.named_parameters():
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
                "context_embedder",
                "x_embedder",
                "transformer_blocks",
                "single_transformer_blocks",
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

        #  As we use Attention from diffusers, it will be automatically initialized.
        for block in self.transformer_blocks:
            for module in [block.attn.norm_q, block.attn.norm_k, block.attn.norm_added_q, block.attn.norm_added_k]:
                module.weight.initialized = True
        for block in self.single_transformer_blocks:
            for module in [block.attn.norm_q, block.attn.norm_k]:
                module.weight.initialized = True

    def load_model(self):
        checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
        if self.cfg.pretrained_source == "flux":
            self.load_state_dict(checkpoint)
        elif self.cfg.pretrained_source == "dc-gen":
            self.load_state_dict(checkpoint[self.cfg.load_type])
        else:
            raise ValueError(f"Pretrained source {self.cfg.pretrained_source} is not supported")

    def prepare_latents_and_ids(self, text_embeddings: torch.Tensor, latents: torch.Tensor):
        device, dtype = text_embeddings.device, text_embeddings.dtype
        text_ids = torch.zeros(text_embeddings.shape[1], 3).to(device=device, dtype=dtype)

        device, dtype = latents.device, latents.dtype
        bsz, num_channels, height, width = latents.shape
        image_ids = torch.zeros(height // self.cfg.patch_size, width // self.cfg.patch_size, 3)
        image_ids[..., 1] = image_ids[..., 1] + torch.arange(height // self.cfg.patch_size)[:, None]
        image_ids[..., 2] = image_ids[..., 2] + torch.arange(width // self.cfg.patch_size)[None, :]

        image_ids = image_ids.reshape((height // self.cfg.patch_size) * (width // self.cfg.patch_size), 3)
        image_ids = image_ids.to(device=device, dtype=dtype)

        latents = latents.view(
            bsz,
            num_channels,
            height // self.cfg.patch_size,
            self.cfg.patch_size,
            width // self.cfg.patch_size,
            self.cfg.patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            bsz,
            (height // self.cfg.patch_size) * (width // self.cfg.patch_size),
            num_channels * self.cfg.patch_size * self.cfg.patch_size,
        )

        return latents, text_ids, image_ids

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
        guidance: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        image_ids: Optional[torch.Tensor] = None,
        training=True,
    ):
        clip_text_embeddings = text_embed_info[self.cfg.clip_text_encoder_id]["text_embeddings"]
        t5_text_embeddings = text_embed_info[self.cfg.t5_text_encoder_id]["text_embeddings"]
        if training:
            batch_size, num_channels, height, width = x.shape
            assert text_ids is None and image_ids is None and guidance is not None
            x, text_ids, image_ids = self.prepare_latents_and_ids(t5_text_embeddings, x)

        x = self.x_embedder(x)

        timestep = timestep.to(x.dtype)
        guidance = guidance.to(x.dtype) * 1000
        temb = self.time_text_embed(timestep, guidance, clip_text_embeddings)

        hidden_states = x
        encoder_hidden_states = self.context_embedder(t5_text_embeddings)

        ids = torch.cat((text_ids, image_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if training:
            output = rearrange(
                output, "b (h w) c -> b c h w", h=height // self.cfg.patch_size, w=width // self.cfg.patch_size
            )
            output = F.pixel_shuffle(output, upscale_factor=self.cfg.patch_size)

        return output

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        text_embeddings = text_embed_info[self.cfg.clip_text_encoder_id]["text_embeddings"]
        text_embeddings_2 = text_embed_info[self.cfg.t5_text_encoder_id]["text_embeddings"]

        device = get_device(self)
        bs = text_embeddings.shape[0]

        if noise is None:
            noise = torch.randn(
                bs, self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size, device=device, generator=generator
            )
            noise = noise.to(dtype=text_embeddings.dtype)

        latents = noise
        guidance = torch.tensor([cfg_scale], device=device)
        guidance = guidance.repeat(bs)
        latents, text_ids, image_ids = self.prepare_latents_and_ids(text_embeddings_2, latents)

        if self.cfg.eval_scheduler == "FluxScheduler":
            timesteps = self.eval_scheduler.get_timesteps(device=device)
            for t in timesteps:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.forward_without_cfg(
                    x=latents,
                    timestep=timestep,
                    text_embed_info=text_embed_info,
                    guidance=guidance,
                    text_ids=text_ids,
                    image_ids=image_ids,
                    training=False,
                )

                latents = self.eval_scheduler.step(noise_pred, t, latents, return_dict=False)
        elif self.cfg.eval_scheduler == "Flux2Scheduler":
            timesteps = self.eval_scheduler.get_timesteps(device=device, image_seq_len=self.cfg.input_size**2)
            for t in timesteps:
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.forward_without_cfg(
                    x=latents,
                    timestep=timestep,
                    text_embed_info=text_embed_info,
                    guidance=guidance,
                    text_ids=text_ids,
                    image_ids=image_ids,
                    training=False,
                )

                latents = self.eval_scheduler.step(noise_pred, t, latents)
        else:
            raise NotImplementedError(f"eval_scheduler {self.cfg.eval_scheduler} is not supported in generate()")

        latents = latents.view(
            bs,
            self.cfg.input_size // self.cfg.patch_size,
            self.cfg.input_size // self.cfg.patch_size,
            self.cfg.in_channels,
            self.cfg.patch_size,
            self.cfg.patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(bs, self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size)

        return latents

    def forward_train(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        from diffusers.training_utils import compute_density_for_timestep_sampling

        info = {}
        detailed_loss_dict = {}
        latents = x
        noise = torch.randn_like(latents)
        bs = latents.shape[0]

        # Stage 1: Compute noisy_latents and timesteps (scheduler-specific)
        if self.cfg.train_scheduler == "FlowMatchEulerDiscreteScheduler":
            image_seq_len = latents.shape[2] * latents.shape[3]
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

            image_seq_len = latents.shape[1] * latents.shape[2] * latents.shape[3]

            indices = (u * self.cfg.train_sampling_steps).long()
            sigmas = (self.cfg.train_sampling_steps - indices) / self.cfg.train_sampling_steps
            # FLUX2 shift: t' = (r * t) / (1 + (r - 1) * t), where r = sqrt(image_seq_len / base_seq_len)
            mu = calculate_shift_flux2(image_seq_len, self.cfg.base_seq_len)
            sigmas = (mu * sigmas) / (1.0 + (mu - 1.0) * sigmas)
            sigmas = sigmas.to(device=latents.device)
            timesteps = sigmas * self.cfg.train_sampling_steps
            sigmas = sigmas.view(bs, *([1] * (latents.ndim - 1)))
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        # Stage 2: Forward pass + loss
        guidance_one = torch.ones((bs,), device=timesteps.device)
        target = noise - latents

        if self.cfg.use_guide_loss:
            cfg_min, cfg_max = 1, 14
            guidance = torch.rand((bs,), device=timesteps.device) * (cfg_max - cfg_min) + cfg_min

            mask = torch.rand(bs, device=latents.device) < self.cfg.drop_text_raio

            clip_null_embedding = self.clip_null_embedding.to(latents)
            t5_null_embedding = self.t5_null_embedding.to(latents)
            text_embed_info[self.cfg.clip_text_encoder_id]["text_embeddings"][mask] = clip_null_embedding
            text_embed_info[self.cfg.t5_text_encoder_id]["text_embeddings"][mask] = t5_null_embedding

            null_text_embeddings = clip_null_embedding.unsqueeze(0).repeat(bs, 1)
            null_text_embeddings_2 = t5_null_embedding.unsqueeze(0).repeat(bs, 1, 1)
            null_text_embed_info = {
                self.cfg.clip_text_encoder_id: {"text_embeddings": null_text_embeddings},
                self.cfg.t5_text_encoder_id: {"text_embeddings": null_text_embeddings_2},
            }

            noise_pred_guided = self.forward_without_cfg(
                noisy_latents, timesteps, text_embed_info=text_embed_info, guidance=guidance
            )

            with torch.no_grad():
                noise_pred_uncond = self.forward_without_cfg(
                    noisy_latents, timesteps, text_embed_info=null_text_embed_info, guidance=guidance_one
                )
            noise_pred = (
                noise_pred_guided - (guidance_one.view(bs, 1, 1, 1) - guidance.view(bs, 1, 1, 1)) * noise_pred_uncond
            ) / guidance.view(bs, 1, 1, 1)
        else:
            noise_pred = self.forward_without_cfg(
                noisy_latents, timesteps, text_embed_info=text_embed_info, guidance=guidance_one
            )

        loss = torch.mean(((noise_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), dim=1)
        loss = loss.mean()

        detailed_loss_dict["loss"] = loss.item()
        info["detailed_loss_dict"] = detailed_loss_dict
        return {0: loss}, info
