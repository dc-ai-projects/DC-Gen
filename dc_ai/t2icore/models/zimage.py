# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...models.utils.network import get_device, get_submodule_weights
from .base_diffusion import BaseT2IDiffusionModel, BaseT2IDiffusionModelConfig

__all__ = ["ZImageConfig", "ZImage"]


@dataclass
class ZImageConfig(BaseT2IDiffusionModelConfig):
    name: str = "ZImage"
    pretrained_path: Optional[str] = None
    pretrained_source: str = "diffusers"
    eval_scheduler: str = "FluxScheduler"
    train_scheduler: str = "FlowMatchEulerDiscreteScheduler"
    num_inference_steps: int = 8  # ZImage-Turbo uses 8 NFE

    # Scheduler settings (from ZImage-Turbo config)
    flow_shift: float = 3.0
    use_dynamic_shifting: bool = False

    # Patch settings
    all_patch_size: tuple[int, ...] = (2,)
    all_f_patch_size: tuple[int, ...] = (1,)
    in_channels: int = 16
    patch_size: int = 2

    # Model dimensions
    dim: int = 3840
    n_layers: int = 30
    n_refiner_layers: int = 2
    n_heads: int = 30
    n_kv_heads: int = 30
    norm_eps: float = 1e-5
    qk_norm: bool = True

    # Text encoder settings
    cap_feat_dim: int = 2560  # Qwen3 output dimension
    text_encoder_id: str = "z-image-turbo/qwen3-4b-512-bf16"

    # RoPE settings
    rope_theta: float = 256.0
    t_scale: float = 1000.0
    axes_dims: tuple[int, ...] = (32, 48, 48)
    axes_lens: tuple[int, ...] = (1536, 512, 512)

    # Training settings
    freeze_backbone: bool = False
    head_only: bool = False
    use_guide_loss: bool = False
    drop_text_ratio: float = 0.0
    null_embeds_dir: Optional[str] = None
    train_cfg_value: float = 3.0  # CFG value for guidance loss during training

    # Flux2 shift settings
    base_seq_len: int = 4096


class ZImage(BaseT2IDiffusionModel):
    def __init__(self, cfg: ZImageConfig):
        super().__init__(cfg)
        self.cfg: ZImageConfig

        if cfg.freeze_backbone or cfg.head_only:
            self.transformer.requires_grad_(False)
            if cfg.head_only:
                unfreeze_modules = self.transformer.all_final_layer.values()
            else:
                unfreeze_modules = (
                    *self.transformer.all_x_embedder.values(),
                    *self.transformer.all_final_layer.values(),
                )

            for module in unfreeze_modules:
                module.requires_grad_(True)

    def build_model(self) -> None:
        from diffusers import ZImageTransformer2DModel

        if self.cfg.use_guide_loss and self.cfg.null_embeds_dir is None:
            raise ValueError("Z-Image guide loss requires null_embeds_dir")
        if self.cfg.drop_text_ratio > 0 and self.cfg.null_embeds_dir is None:
            raise ValueError("Z-Image text dropout requires null_embeds_dir")

        def build_transformer() -> ZImageTransformer2DModel:
            return ZImageTransformer2DModel(
                all_patch_size=self.cfg.all_patch_size,
                all_f_patch_size=self.cfg.all_f_patch_size,
                in_channels=self.cfg.in_channels,
                dim=self.cfg.dim,
                n_layers=self.cfg.n_layers,
                n_refiner_layers=self.cfg.n_refiner_layers,
                n_heads=self.cfg.n_heads,
                n_kv_heads=self.cfg.n_kv_heads,
                norm_eps=self.cfg.norm_eps,
                qk_norm=self.cfg.qk_norm,
                cap_feat_dim=self.cfg.cap_feat_dim,
                rope_theta=self.cfg.rope_theta,
                t_scale=self.cfg.t_scale,
                axes_dims=self.cfg.axes_dims,
                axes_lens=self.cfg.axes_lens,
            )

        if self.cfg.pretrained_path is None:
            self.transformer = build_transformer()
        else:
            # Assign normalized weights into a meta model without materializing a second full transformer.
            with torch.device("meta"):
                self.transformer = build_transformer()

        if self.cfg.null_embeds_dir:
            import os

            qwen_null_embedding_path = os.path.join(self.cfg.null_embeds_dir, f"{self.cfg.text_encoder_id}.pth")
            qwen_null_embedding = torch.load(qwen_null_embedding_path, weights_only=True, map_location="cpu")
            # qwen_null_embedding: [num_valid_tokens, dim] e.g. [8, 2560]
            # Used as-is for inference (variable-length cap_feats_list)
            self.register_buffer("qwen_null_embedding", qwen_null_embedding, persistent=False)
            # Also create a padded version [text_max_length, dim] for training
            # (training text_embeddings are padded to fixed length, e.g. 512)
            text_max_length = 512  # matches tokenizer max_length used in data pipeline
            if qwen_null_embedding.dim() == 2 and qwen_null_embedding.shape[0] < text_max_length:
                pad_len = text_max_length - qwen_null_embedding.shape[0]
                qwen_null_embedding_padded = F.pad(qwen_null_embedding, (0, 0, 0, pad_len), value=0)
            else:
                qwen_null_embedding_padded = qwen_null_embedding
            self.register_buffer("qwen_null_embedding_padded", qwen_null_embedding_padded, persistent=False)

    def _prepare_channel_mask(
        self,
        x: torch.Tensor,
        channel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Validate and normalize an original-channel mask for latents shaped [B, C, H, W]."""
        if x.ndim != 4:
            raise ValueError(f"Z-Image latents must have shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.cfg.in_channels:
            raise ValueError(f"Z-Image latents must have {self.cfg.in_channels} channels, got {x.shape[1]}")
        if channel_mask is None:
            return torch.ones(
                (x.shape[0], x.shape[1]),
                dtype=torch.bool,
                device=x.device,
            )
        if channel_mask.dtype != torch.bool:
            raise TypeError(f"Z-Image channel_mask must have dtype bool, got {channel_mask.dtype}")
        if channel_mask.ndim != 2 or channel_mask.shape != x.shape[:2]:
            raise ValueError(
                f"Z-Image channel_mask must have shape {tuple(x.shape[:2])}, got {tuple(channel_mask.shape)}"
            )
        if not channel_mask.any(dim=1).all():
            raise ValueError("Z-Image channel_mask must activate at least one channel for every sample")
        return channel_mask.to(device=x.device)

    @staticmethod
    def _apply_channel_mask(
        x: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        return x * channel_mask[:, :, None, None].to(dtype=x.dtype)

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (prediction.float() - target.float()) ** 2
        latent_mask = channel_mask[:, :, None, None].to(dtype=squared_error.dtype)
        active_elements = channel_mask.sum(dim=1).to(dtype=squared_error.dtype) * target.shape[2] * target.shape[3]
        per_sample_loss = (squared_error * latent_mask).sum(dim=(1, 2, 3)) / active_elements
        return per_sample_loss.mean()

    def _extract_cap_feats(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
    ) -> list[torch.Tensor]:
        encoder_info = text_embed_info[self.cfg.text_encoder_id]
        embeddings = encoder_info["text_embeddings"]
        attention_mask = encoder_info["attention_mask"]
        return [
            sample_embeddings[sample_mask.bool()]
            for sample_embeddings, sample_mask in zip(embeddings, attention_mask, strict=True)
        ]

    def _replace_dropped_text_with_null(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        drop_mask: torch.Tensor,
        latents: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        text_embeddings = text_embed_info[self.cfg.text_encoder_id]["text_embeddings"]
        attention_mask = text_embed_info[self.cfg.text_encoder_id]["attention_mask"]

        if drop_mask.any():
            text_embeddings = text_embeddings.clone()
            attention_mask = attention_mask.clone()
            num_dropped = int(drop_mask.sum().item())
            null_valid_len = self.qwen_null_embedding.shape[0]
            null_slice = self.qwen_null_embedding_padded[: text_embeddings.shape[1]].to(latents)
            text_embeddings[drop_mask] = null_slice.unsqueeze(0).expand(num_dropped, -1, -1)

            null_attention_mask = torch.zeros(
                attention_mask.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            null_attention_mask[:null_valid_len] = 1
            attention_mask[drop_mask] = null_attention_mask.unsqueeze(0).expand(num_dropped, -1)

        conditional_text_embed_info = {
            self.cfg.text_encoder_id: {
                "text_embeddings": text_embeddings,
                "attention_mask": attention_mask,
            },
        }
        return conditional_text_embed_info

    def _get_null_text_embed_info(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        batch_size: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        attention_mask = text_embed_info[self.cfg.text_encoder_id]["attention_mask"]
        null_text_embeddings = self.qwen_null_embedding_padded.unsqueeze(0).repeat(batch_size, 1, 1)
        null_attention_mask = torch.zeros(
            (batch_size, null_text_embeddings.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        null_attention_mask[:, : self.qwen_null_embedding.shape[0]] = 1
        return {
            self.cfg.text_encoder_id: {
                "text_embeddings": null_text_embeddings,
                "attention_mask": null_attention_mask,
            }
        }

    def initialize_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""

        if self.cfg.pretrained_path is not None:
            for parameter in self.parameters():
                parameter.initialized = True
            return

        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.weight.initialized = True
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    module.bias.initialized = True

        self.apply(_basic_init)

        nn.init.normal_(self.transformer.x_pad_token, std=0.02)
        self.transformer.x_pad_token.initialized = True
        nn.init.normal_(self.transformer.cap_pad_token, std=0.02)
        self.transformer.cap_pad_token.initialized = True

        # Mark all RMSNorm and LayerNorm weights as initialized
        # (they are auto-initialized by diffusers or PyTorch)
        for module in self.modules():
            if "RMSNorm" in module.__class__.__name__ or isinstance(module, nn.LayerNorm):
                if hasattr(module, "weight") and module.weight is not None:
                    module.weight.initialized = True
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.initialized = True

    def load_model(self) -> None:
        if self.cfg.pretrained_path is None:
            raise ValueError("Z-Image loading requires pretrained_path")

        checkpoint = torch.load(
            self.cfg.pretrained_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"Z-Image checkpoint must contain a state dict: {self.cfg.pretrained_path}")

        if self.cfg.pretrained_source == "diffusers":
            state_dict = checkpoint
        elif self.cfg.pretrained_source == "dc-gen":
            model_state_dict = checkpoint.get("model_state_dict")
            if not isinstance(model_state_dict, Mapping):
                raise ValueError(
                    "Z-Image DC-Gen checkpoint must contain model_state_dict: " f"{self.cfg.pretrained_path}"
                )
            state_dict = get_submodule_weights(model_state_dict, "transformer.")
            if not state_dict:
                raise ValueError(
                    "Z-Image DC-Gen checkpoint model_state_dict must contain transformer.* weights: "
                    f"{self.cfg.pretrained_path}"
                )
        else:
            raise ValueError(f"Unsupported Z-Image pretrained source {self.cfg.pretrained_source!r}")

        non_tensor_keys = [key for key, value in state_dict.items() if not isinstance(value, torch.Tensor)]
        if non_tensor_keys:
            raise ValueError(f"Z-Image checkpoint contains non-tensor values: {non_tensor_keys[:20]}")

        self.transformer.load_state_dict(state_dict, strict=True, assign=True)

    def get_lora_target_modules(self) -> set[str]:
        """Get target modules for LoRA adaptation."""
        target_modules_name_list = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "w1",
            "w2",
            "w3",
        ]
        target_modules = set()
        for name, module in self.transformer.named_modules():
            if isinstance(module, nn.Linear):
                for target in target_modules_name_list:
                    if target in name:
                        target_modules.add(name)
        return target_modules

    def get_trainable_modules_list(self) -> nn.ModuleList:
        return nn.ModuleList([self.transformer])

    def enable_activation_checkpointing(self, mode: str) -> None:
        if mode == "transformer":
            self.transformer.enable_gradient_checkpointing()
        else:
            self.transformer.disable_gradient_checkpointing()

    def forward_without_cfg(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        guidance: Optional[torch.Tensor] = None,
        training=True,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass compatible with dc-gen framework.

        Args:
            x: Batched input latents [B, C, H, W]
            timestep: Timestep tensor [B]
            text_embed_info: Dict containing text embeddings
            guidance: Guidance scale (not used for ZImage-Turbo)
            training: Whether in training mode
            channel_mask: Original latent-channel mask shaped [B, C]

        Returns:
            Predicted noise/velocity [B, C, H, W]
        """
        del guidance, training

        channel_mask = self._prepare_channel_mask(x, channel_mask)
        x = self._apply_channel_mask(x, channel_mask)

        cap_feats_list = self._extract_cap_feats(text_embed_info)

        # Convert batched tensor to list format for internal processing
        # Add F (frame) dimension for image: [B, C, H, W] -> List of [C, 1, H, W]
        bsz = x.shape[0]
        x_list = [x[i].unsqueeze(1) for i in range(bsz)]  # Add F=1 dimension

        output_list = self.transformer(
            x=x_list,
            t=timestep,
            cap_feats=cap_feats_list,
            patch_size=self.cfg.patch_size,
            f_patch_size=1,
            return_dict=False,
        )[0]

        # Convert list back to batched tensor
        # Remove F dimension: List of [C, 1, H, W] -> [B, C, H, W]
        output = torch.stack([out.squeeze(1) for out in output_list], dim=0)

        return self._apply_channel_mask(output, channel_mask)

    @torch.no_grad()
    def generate(
        self,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        noise: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        pag_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        negative_text_embed_info: Optional[dict[str, dict[str, torch.Tensor]]] = None,
        cfg_normalization: bool = False,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate images from text embeddings.

        Args:
            text_embed_info: Dict containing positive text embeddings
            noise: Initial noise tensor [B, C, H, W]
            cfg_scale: Classifier-free guidance scale (0.0 for ZImage-Turbo, 4.0+ for ZImage-Base)
            pag_scale: PAG scale (not used)
            generator: Random generator
            negative_text_embed_info: Dict containing negative text embeddings (for CFG)
            cfg_normalization: Whether to apply CFG normalization (from diffusers)
            channel_mask: Original latent-channel mask shaped [B, C]

        Returns:
            Generated latents [B, C, H, W]
        """
        del pag_scale

        qwen_embeddings = text_embed_info[self.cfg.text_encoder_id]["text_embeddings"]
        cap_feats_list = self._extract_cap_feats(text_embed_info)

        # Prepare negative embeddings if CFG is enabled
        do_classifier_free_guidance = cfg_scale > 1.0
        negative_cap_feats_list = None
        if do_classifier_free_guidance:
            if negative_text_embed_info is None:
                # Use null embedding if negative_text_embed_info not provided
                if hasattr(self, "qwen_null_embedding"):
                    bs = qwen_embeddings.shape[0]
                    # Create separate null embedding for each sample (same content but separate tensors)
                    negative_cap_feats_list = [self.qwen_null_embedding.clone() for _ in range(bs)]
                else:
                    raise ValueError("CFG requires null_embeds_dir or negative_text_embed_info")
            else:
                # Use provided negative embeddings
                negative_cap_feats_list = self._extract_cap_feats(negative_text_embed_info)

        device = get_device(self)
        bs = qwen_embeddings.shape[0]
        expected_shape = (bs, self.cfg.in_channels, self.cfg.input_size, self.cfg.input_size)

        if noise is None:
            latents = torch.randn(
                expected_shape,
                device=device,
                generator=generator,
                dtype=torch.float32,
            )
        else:
            if tuple(noise.shape) != expected_shape:
                raise ValueError(f"Z-Image noise must have shape {expected_shape}, got {tuple(noise.shape)}")
            latents = noise.to(device=device, dtype=torch.float32)

        # ZImage keeps latents in float32 throughout
        channel_mask = self._prepare_channel_mask(latents, channel_mask)
        latents = self._apply_channel_mask(latents, channel_mask)

        if self.cfg.eval_scheduler == "FluxScheduler":
            timesteps = self.eval_scheduler.get_timesteps(device=device)
            scheduler_step_kwargs = {"return_dict": False}
        elif self.cfg.eval_scheduler == "Flux2Scheduler":
            active_channels = channel_mask.sum(dim=1)
            if not torch.equal(active_channels, active_channels[:1].expand_as(active_channels)):
                raise ValueError(
                    "Z-Image Flux2Scheduler inference requires every sample to have the same number of active channels"
                )
            image_seq_len = int((active_channels[0] * latents.shape[2] * latents.shape[3]).item())
            timesteps = self.eval_scheduler.get_timesteps(device=device, image_seq_len=image_seq_len)
            scheduler_step_kwargs = {}
        else:
            raise NotImplementedError(f"eval_scheduler {self.cfg.eval_scheduler} is not supported in generate()")

        for timestep in timesteps:
            noise_pred = self._denoise_step(
                timestep,
                latents,
                bs,
                apply_cfg=do_classifier_free_guidance,
                cap_feats_list=cap_feats_list,
                negative_cap_feats_list=negative_cap_feats_list,
                cfg_scale=cfg_scale,
                cfg_normalization=cfg_normalization,
                channel_mask=channel_mask,
            )
            latents = self.eval_scheduler.step(
                noise_pred.to(torch.float32),
                timestep,
                latents,
                **scheduler_step_kwargs,
            )
            latents = self._apply_channel_mask(latents, channel_mask)
            assert latents.dtype == torch.float32

        return latents

    def _denoise_step(
        self,
        t: torch.Tensor,
        latents: torch.Tensor,
        bs: int,
        apply_cfg: bool,
        cap_feats_list: list[torch.Tensor],
        negative_cap_feats_list: Optional[list[torch.Tensor]],
        cfg_scale: float,
        cfg_normalization: bool,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        timestep = t.expand(bs)
        timestep = (1000 - timestep) / 1000
        timestep = timestep.to(latents.dtype)

        model_dtype = next(self.parameters()).dtype
        latents_typed = latents.to(model_dtype)
        latents_typed = self._apply_channel_mask(latents_typed, channel_mask)

        if apply_cfg:
            if negative_cap_feats_list is None:
                raise ValueError("Z-Image CFG denoising requires negative caption features")
            latent_model_input = latents_typed.repeat(2, 1, 1, 1)
            latent_model_input_list = [latent_model_input[j].unsqueeze(1) for j in range(2 * bs)]
            cap_feats_combined = cap_feats_list + negative_cap_feats_list
            timestep_input = timestep.repeat(2)
            model_out_list = self.transformer(
                x=latent_model_input_list,
                t=timestep_input,
                cap_feats=cap_feats_combined,
                patch_size=self.cfg.patch_size,
                f_patch_size=1,
                return_dict=False,
            )[0]
            pos_out = model_out_list[:bs]
            neg_out = model_out_list[bs:]
            noise_pred = []
            cfg_normalization_scale = float(cfg_normalization)
            for j in range(bs):
                pos = pos_out[j].squeeze(1).float()
                neg = neg_out[j].squeeze(1).float()
                pred = pos + cfg_scale * (pos - neg)
                if cfg_normalization_scale > 0.0:
                    ori_pos_norm = torch.linalg.vector_norm(pos)
                    new_pos_norm = torch.linalg.vector_norm(pred)
                    max_new_norm = ori_pos_norm * cfg_normalization_scale
                    if new_pos_norm > max_new_norm:
                        pred = pred * (max_new_norm / new_pos_norm)
                noise_pred.append(pred)
            noise_pred = torch.stack(noise_pred, dim=0)
        else:
            latents_list = [latents_typed[j].unsqueeze(1) for j in range(bs)]
            noise_pred_list = self.transformer(
                x=latents_list,
                t=timestep,
                cap_feats=cap_feats_list,
                patch_size=self.cfg.patch_size,
                f_patch_size=1,
                return_dict=False,
            )[0]
            noise_pred = torch.stack([pred.squeeze(1) for pred in noise_pred_list], dim=0)

        return -self._apply_channel_mask(noise_pred, channel_mask)

    def forward_train(
        self,
        x: torch.Tensor,
        text_embed_info: dict[str, dict[str, torch.Tensor]],
        generator: Optional[torch.Generator] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[int, torch.Tensor], dict]:
        """Training forward pass with FlowMatch loss.

        Args:
            x: Clean latents [B, C, H, W]
            text_embed_info: Dict containing text embeddings
            generator: Random generator
            channel_mask: Original latent-channel mask shaped [B, C]

        Returns:
            Tuple of (loss_dict, info_dict)
        """
        del generator

        from diffusers.training_utils import compute_density_for_timestep_sampling

        channel_mask = self._prepare_channel_mask(x, channel_mask)
        latents = self._apply_channel_mask(x, channel_mask)
        noise = self._apply_channel_mask(torch.randn_like(latents), channel_mask)
        batch_size = latents.shape[0]

        if self.cfg.train_scheduler == "FlowMatchEulerDiscreteScheduler":
            from .flux import calculate_shift

            image_seq_len = latents.shape[2] * latents.shape[3]
            self.train_scheduler.set_timesteps(self.cfg.train_sampling_steps, mu=calculate_shift(image_seq_len))

            timestep_density = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=None,
            )
            timestep_indices = (timestep_density * self.train_scheduler.config.num_train_timesteps).long()
            timesteps = self.train_scheduler.timesteps[timestep_indices].to(device=latents.device)
            noisy_latents = self.train_scheduler.scale_noise(latents, timesteps, noise)
        elif self.cfg.train_scheduler == "FlowMatchEulerDiscreteSchedulerFlux2":
            timestep_density = compute_density_for_timestep_sampling(
                weighting_scheme="logit_normal",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
            )
            timestep_indices = (timestep_density * self.cfg.train_sampling_steps).long()
            sigmas = ((self.cfg.train_sampling_steps - timestep_indices) / self.cfg.train_sampling_steps).to(
                device=latents.device
            )
            active_latent_numel = channel_mask.sum(dim=1).to(dtype=torch.float32) * latents.shape[2] * latents.shape[3]
            shift_ratio = torch.sqrt(active_latent_numel / self.cfg.base_seq_len)
            sigmas = (shift_ratio * sigmas) / (1.0 + (shift_ratio - 1.0) * sigmas)
            timesteps = sigmas * self.cfg.train_sampling_steps
            sigma_broadcast = sigmas.view(batch_size, *([1] * (latents.ndim - 1)))
            noisy_latents = sigma_broadcast * noise + (1.0 - sigma_broadcast) * latents
        else:
            raise NotImplementedError(f"train scheduler {self.cfg.train_scheduler} is not supported")

        noisy_latents = self._apply_channel_mask(noisy_latents, channel_mask)
        timesteps_normalized = (1000 - timesteps) / 1000

        if self.cfg.use_guide_loss:
            if not hasattr(self, "qwen_null_embedding_padded"):
                raise RuntimeError("Z-Image guide loss requires loaded null embeddings")
            guidance = torch.full(
                (batch_size,),
                self.cfg.train_cfg_value,
                device=timesteps.device,
                dtype=torch.float32,
            )
            guidance_one = torch.ones((batch_size,), device=timesteps.device)
            drop_mask = torch.rand(batch_size, device=latents.device) < self.cfg.drop_text_ratio
            conditional_text_embed_info = self._replace_dropped_text_with_null(
                text_embed_info,
                drop_mask,
                latents,
            )
            null_text_embed_info = self._get_null_text_embed_info(text_embed_info, batch_size)
            noise_pred_guided = self.forward_without_cfg(
                noisy_latents,
                timesteps_normalized,
                text_embed_info=conditional_text_embed_info,
                guidance=guidance,
                training=True,
                channel_mask=channel_mask,
            )
            with torch.no_grad():
                noise_pred_uncond = self.forward_without_cfg(
                    noisy_latents,
                    timesteps_normalized,
                    text_embed_info=null_text_embed_info,
                    guidance=guidance_one,
                    training=True,
                    channel_mask=channel_mask,
                )
            guidance_broadcast = guidance.view(batch_size, *([1] * (latents.ndim - 1)))
            noise_pred = (noise_pred_guided - (1.0 - guidance_broadcast) * noise_pred_uncond) / guidance_broadcast
        else:
            conditional_text_embed_info = text_embed_info
            if self.cfg.drop_text_ratio > 0:
                drop_mask = torch.rand(batch_size, device=latents.device) < self.cfg.drop_text_ratio
                conditional_text_embed_info = self._replace_dropped_text_with_null(
                    text_embed_info,
                    drop_mask,
                    latents,
                )
            noise_pred = self.forward_without_cfg(
                noisy_latents,
                timesteps_normalized,
                text_embed_info=conditional_text_embed_info,
                training=True,
                channel_mask=channel_mask,
            )

        target = latents - noise
        loss = self._masked_mse(noise_pred, target, channel_mask)
        return {0: loss}, {"detailed_loss_dict": {"loss": loss.item()}}
