import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.utils import USE_PEFT_BACKEND, is_torch_npu_available, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from far.utils.registry import MODEL_REGISTRY

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register()
class DCGenFluxModel(FluxTransformer2DModel):
    pass


class FluxTwoTimestepTextProjEmbeddings(nn.Module):

    def __init__(self, embedding_dim, pooled_projection_dim, gate_value=0.0, deltatime_type='r', guidance_embed=False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.delta_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.guidance_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim) if guidance_embed else None
        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, embedding_dim, act_fn="silu")

        self.register_buffer('delta_emb_gate', torch.tensor([gate_value], dtype=torch.float32), persistent=False)
        self.deltatime_type = deltatime_type
        self.gate_track = gate_value

    def forward(self, timestep, r_timestep, pooled_projection, guidance=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=pooled_projection.dtype))

        if self.deltatime_type == 'r':
            delta_input = r_timestep
        elif self.deltatime_type == 't-r':
            delta_input = timestep - r_timestep
        else:
            raise ValueError(f"Unknown deltatime_type: {self.deltatime_type}")

        delta_proj = self.time_proj(delta_input)
        delta_emb = self.delta_embedder(delta_proj.to(dtype=pooled_projection.dtype))

        gate = self.delta_emb_gate.to(dtype=pooled_projection.dtype)
        self.gate_track = float(gate)

        temb = (1 - gate) * timesteps_emb + gate * delta_emb
        if self.guidance_embedder is not None:
            if guidance is None:
                guidance = torch.ones_like(timestep)
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(guidance_proj.to(dtype=pooled_projection.dtype))
            temb = temb + guidance_emb

        pooled_projections = self.text_embedder(pooled_projection)
        return temb + pooled_projections


@MODEL_REGISTRY.register()
class DCGenFluxFlowMapModel(FluxTransformer2DModel):

    def setup_flowmap_model(self, gate_value=0.0, deltatime_type='r'):
        orig = self.time_text_embed
        new_embed = FluxTwoTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
            guidance_embed=hasattr(orig, "guidance_embedder"),
        )
        new_embed.time_proj = copy.deepcopy(orig.time_proj)
        new_embed.timestep_embedder = copy.deepcopy(orig.timestep_embedder)
        new_embed.delta_embedder = copy.deepcopy(orig.timestep_embedder)
        if hasattr(orig, "guidance_embedder") and new_embed.guidance_embedder is not None:
            new_embed.guidance_embedder = copy.deepcopy(orig.guidance_embedder)
        new_embed.text_embedder = copy.deepcopy(orig.text_embedder)
        del self.time_text_embed
        self.time_text_embed = new_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        r_timestep: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        if r_timestep is not None:
            r_timestep = r_timestep.to(hidden_states.dtype) * 1000
            temb = self.time_text_embed(timestep, r_timestep, pooled_projections, guidance=guidance)
        else:
            temb = self.time_text_embed(timestep, timestep, pooled_projections, guidance=guidance)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        if is_torch_npu_available():
            freqs_cos, freqs_sin = self.pos_embed(ids.cpu())
            image_rotary_emb = (freqs_cos.npu(), freqs_sin.npu())
        else:
            image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                    temb=temb, image_rotary_emb=image_rotary_emb, joint_attention_kwargs=joint_attention_kwargs,
                )

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
