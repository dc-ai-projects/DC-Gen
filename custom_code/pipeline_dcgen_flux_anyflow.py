from contextlib import nullcontext
from typing import List, Optional, Union
import time

import torch
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderDC, FluxTransformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from far.schedulers.scheduling_flowmap_euler_discrete import FlowMapDiscreteScheduler
from far.utils.registry import PIPELINE_REGISTRY

logger = logging.get_logger(__name__)


@PIPELINE_REGISTRY.register()
class DCGenFluxAnyFlowPipeline(DiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderDC,
        tokenizer: CLIPTokenizer,
        tokenizer_2: T5TokenizerFast,
        text_encoder: CLIPTextModel,
        text_encoder_2: T5EncoderModel,
        transformer: FluxTransformer2DModel,
        scheduler: FlowMapDiscreteScheduler,
        null_prompt_embeds: Optional[torch.Tensor] = None,
        null_pooled_prompt_embeds: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae, tokenizer=tokenizer, tokenizer_2=tokenizer_2,
            text_encoder=text_encoder, text_encoder_2=text_encoder_2,
            transformer=transformer, scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.null_prompt_embeds = null_prompt_embeds
        self.null_pooled_prompt_embeds = null_pooled_prompt_embeds

    @torch.no_grad()
    def encode_prompt(self, prompt, device, max_sequence_length=512):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        clip_inputs = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt',
        )
        pooled_prompt_embeds = self.text_encoder(
            clip_inputs.input_ids.to(device), output_hidden_states=False
        ).pooler_output

        t5_inputs = self.tokenizer_2(
            prompt, padding='max_length', max_length=max_sequence_length,
            truncation=True, return_tensors='pt',
        )
        prompt_embeds = self.text_encoder_2(
            t5_inputs.input_ids.to(device), output_hidden_states=False
        )[0]

        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
        return prompt_embeds, pooled_prompt_embeds, txt_ids

    @staticmethod
    def _prepare_latent_image_ids(h, w, device, dtype):
        img_ids = torch.zeros(h, w, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h, device=device)[:, None]
        img_ids[..., 2] = torch.arange(w, device=device)[None, :]
        return img_ids.reshape(h * w, 3)

    def _cache_context(self, key: str):
        cache_context = getattr(self.transformer, 'cache_context', None)
        if callable(cache_context):
            return cache_context(key)

        wrapped_module = getattr(self.transformer, 'module', None)
        cache_context = getattr(wrapped_module, 'cache_context', None)
        if callable(cache_context):
            return cache_context(key)

        return nullcontext()

    def _build_guidance(self, batch_size, device, dtype, guidance_scale):
        if not getattr(self.transformer.config, 'guidance_embeds', False):
            return None

        return torch.full((batch_size,), float(guidance_scale), device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents):
        b, c, h, w = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(b, h * w, c), h, w

    @staticmethod
    def _unpack_latents(latents, h, w):
        b, _, c = latents.shape
        return latents.view(b, h, w, c).permute(0, 3, 1, 2)

    def training_rollout(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        num_inference_steps: int = 4,
        grad_timestep: int = None,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        packed, h, w = self._pack_latents(latents)
        batch_size = latents.shape[0]
        device = latents.device
        N = self.scheduler.config.num_train_timesteps

        img_ids = self._prepare_latent_image_ids(h, w, device, prompt_embeds.dtype)
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=prompt_embeds.dtype)
        guidance = self._build_guidance(batch_size, device, packed.dtype, guidance_scale)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        def inference_range(lat, ts):
            if len(ts) <= 1:
                return lat
            for i in range(len(ts) - 1):
                t, r = ts[i], ts[i + 1]
                if torch.equal(t, r):
                    continue
                t_input = t.expand(batch_size) / N
                r_input = r.expand(batch_size) / N
                with self._cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=lat, timestep=t_input, r_timestep=r_input,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds,
                        img_ids=img_ids, txt_ids=txt_ids, return_dict=False,
                    )[0]
                lat = self.scheduler.step(noise_pred, lat, t, r)
            return lat

        if grad_timestep is None:
            with torch.no_grad():
                packed = inference_range(packed, timesteps)
            return self._unpack_latents(packed, h, w)

        prev_ts = [timesteps[0], timesteps[grad_timestep]]
        curr_ts = [timesteps[grad_timestep], timesteps[grad_timestep + 1]]
        post_ts = [timesteps[grad_timestep + 1], timesteps[-1]]

        packed = inference_range(packed, prev_ts)

        packed = inference_range(packed, curr_ts)

        packed = inference_range(packed, post_ts)

        return self._unpack_latents(packed, h, w)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        output_type: str = 'pil',
        timing_log: Optional[list] = None,
        timing_event=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=None,
    ) -> FluxPipelineOutput:
        device = self._execution_device
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        _t0 = time.perf_counter()

        def _tlog(msg):
            print(msg)
            if timing_log is not None:
                timing_log.append(msg)
            if timing_event is not None:
                timing_event.put('update')

        # --- Stage 1: condition encoding ---
        prompt_embeds, pooled_prompt_embeds, txt_ids = self.encode_prompt(prompt, device)
        do_true_cfg = true_cfg_scale is not None and true_cfg_scale > 1.0
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_true_cfg:
            if negative_prompt is not None:
                if isinstance(negative_prompt, str) and batch_size > 1:
                    negative_prompt = [negative_prompt] * batch_size
                negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.encode_prompt(negative_prompt, device)
            elif self.null_prompt_embeds is not None and self.null_pooled_prompt_embeds is not None:
                negative_prompt_embeds = self.null_prompt_embeds.expand(batch_size, -1, -1).to(
                    device=device, dtype=prompt_embeds.dtype
                )
                negative_pooled_prompt_embeds = self.null_pooled_prompt_embeds.expand(batch_size, -1).to(
                    device=device, dtype=pooled_prompt_embeds.dtype
                )
            else:
                negative_prompt_embeds, negative_pooled_prompt_embeds, _ = self.encode_prompt([""] * batch_size, device)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _t1 = time.perf_counter()
        _tlog(f'[Timing] Condition encoding : {_t1 - _t0:.3f}s  (total {_t1 - _t0:.3f}s)')

        # --- Stage 2: denoising ---
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        num_channels = self.transformer.config.in_channels
        shape = (batch_size, num_channels, latent_h, latent_w)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        latents, h, w = self._pack_latents(latents)

        img_ids = self._prepare_latent_image_ids(h, w, device, prompt_embeds.dtype)
        guidance = self._build_guidance(batch_size, device, latents.dtype, guidance_scale)

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for i in tqdm(range(len(timesteps) - 1), desc="Denoising"):
            t = timesteps[i]
            r = timesteps[i + 1]
            N = self.scheduler.config.num_train_timesteps

            t_input = t.expand(batch_size) / N
            r_input = r.expand(batch_size) / N

            with self._cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=t_input,
                    r_timestep=r_input,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    return_dict=False,
                )[0]
            if do_true_cfg:
                with self._cache_context("uncond"):
                    noise_pred_uncond = self.transformer(
                        hidden_states=latents,
                        timestep=t_input,
                        r_timestep=r_input,
                        guidance=guidance,
                        encoder_hidden_states=negative_prompt_embeds,
                        pooled_projections=negative_pooled_prompt_embeds,
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        return_dict=False,
                    )[0]
                noise_pred = noise_pred_uncond + true_cfg_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, latents, t, r)

            if callback_on_step_end is not None:
                callback_on_step_end(self, i, t, {'latents': latents})

        latents = self._unpack_latents(latents, h, w)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _t2 = time.perf_counter()
        _tlog(f'[Timing] Denoising          : {_t2 - _t1:.3f}s  (total {_t2 - _t0:.3f}s)')

        if output_type == 'latent':
            return FluxPipelineOutput(images=latents)

        # --- Stage 3: VAE decoding ---
        scaling_factor = getattr(self.vae.config, 'scaling_factor', 1.0)
        shift_factor = getattr(self.vae.config, 'shift_factor', 0.0)
        latents = (latents / scaling_factor) + shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        _t3 = time.perf_counter()
        _tlog(f'[Timing] VAE decoding       : {_t3 - _t2:.3f}s  (total {_t3 - _t0:.3f}s)')

        return FluxPipelineOutput(images=image)
