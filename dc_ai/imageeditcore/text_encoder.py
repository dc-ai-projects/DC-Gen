# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import List, Optional, Union

import torch

from ..apps.text_encoder import BaseSingleTextEncoder, BaseTextEncoder


class ImageEditCoreSingleTextEncoder(BaseSingleTextEncoder):
    def build_text_encoder(self):
        if self.id == "qwen-image-edit/qwen2.5-vl-bf16":
            from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

            self.encoder_dtype = torch.bfloat16
            pipeline = "Qwen/Qwen-Image-Edit"
            text_encoder_subfolder = "text_encoder"
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                pipeline,
                subfolder="tokenizer",
            )

            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pipeline,
                subfolder=text_encoder_subfolder,
                torch_dtype=self.encoder_dtype,
            )

            # Convert dtype to align with qwenimage pipeline
            self.text_encoder.model.visual.rotary_pos_emb.inv_freq = (
                self.text_encoder.model.visual.rotary_pos_emb.inv_freq.to(torch.bfloat16)
            )
            self.text_encoder.model.language_model.rotary_emb.inv_freq = (
                self.text_encoder.model.language_model.rotary_emb.inv_freq.to(torch.bfloat16)
            )

            self.processor = Qwen2VLProcessor.from_pretrained(
                pipeline,
                subfolder="processor",
                torch_dtype=self.encoder_dtype,
            )

        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    @torch.no_grad()
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    @torch.no_grad()
    def _get_qwen_prompt_embeds(
        self,
        prompts: Union[str, List[str]] = None,
        images: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompts = [prompts] if isinstance(prompts, str) else prompts

        template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        drop_idx = 64
        text = [template.format(e) for e in prompts]

        model_inputs = self.processor(
            text=text,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(device)

        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    @torch.no_grad()
    def get_text_embed_info(
        self,
        prompts: list[str] | str,
        images: torch.Tensor,
        device: torch.device,
    ):
        if self.id == "qwen-image-edit/qwen2.5-vl-bf16":
            prompts = [prompts] if isinstance(prompts, str) else prompts
            batch_size = len(prompts)

            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompts, images, device)

            _, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size, seq_len)

            return {
                "text_embeddings": prompt_embeds,
                "text_embedding_masks": prompt_embeds_mask,
            }

        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    @torch.no_grad()
    def get_null_embeddings(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if self.id == "qwen-image-edit/qwen2.5-vl-bf16":
            raise ValueError(f"Text encoder ID {self.id} does not support null embeddings")

        null_tokens = self.tokenizer(
            "",
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        null_embeddings = self.text_encoder(null_tokens.input_ids, null_tokens.attention_mask)
        if self.require_state_type == "last_hidden_state":
            null_text_embeddings = null_embeddings.last_hidden_state
        elif self.require_state_type == "hidden_states":
            null_text_embeddings = null_embeddings.hidden_states[1:]
            null_text_embeddings = torch.stack(null_text_embeddings, dim=0)
        elif self.require_state_type == "pooler_output":
            null_text_embeddings = null_embeddings.pooler_output
        elif self.require_state_type == "text_embeds":
            null_text_embeddings = null_embeddings.text_embeds
        else:
            raise NotImplementedError(f"State type {self.require_state_type} is not defined")

        return null_text_embeddings


class ImageEditCoreTextEncoder(BaseTextEncoder):
    def __init__(self, text_encoder_ids: Sequence[str]):
        super().__init__(text_encoder_ids)
        self.text_encoder_list: list[ImageEditCoreSingleTextEncoder]

    def build_single_text_encoder(self, text_encoder_id: str) -> ImageEditCoreSingleTextEncoder:
        return ImageEditCoreSingleTextEncoder(text_encoder_id)

    @torch.no_grad()
    def get_text_embed_info(self, prompts: list[str] | str, images: torch.Tensor, device: torch.device):
        text_embed_info = {}
        for text_encoder in self.text_encoder_list:
            text_embed_info[text_encoder.id] = text_encoder.get_text_embed_info(prompts, images, device)
        return text_embed_info
