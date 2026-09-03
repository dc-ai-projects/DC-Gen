# Here we use text encoder wrapper code from several upstream projects.
# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/builder.py and https://github.com/NVlabs/Sana/blob/main/app/sana_pipeline.py.
# FLUX.1-dev was introduced by Black Forest Lab.
# The original implementation is by Black Forest Lab, licensed under the Apache License 2.0. See https://github.com/black-forest-labs/flux/blob/802fb4713906133fcbd0d8dc5351620ca4773036/src/flux/modules/conditioner.py.
# The Z-Image text encoding implementation is adapted from ZImagePipeline by the Alibaba Z-Image Team and The HuggingFace Team, Copyright 2025, licensed under the Apache License 2.0. See https://github.com/huggingface/diffusers/blob/v0.38.0/src/diffusers/pipelines/z_image/pipeline_z_image.py.

from collections.abc import Sequence

import torch

from ..apps.text_encoder import BaseSingleTextEncoder, BaseTextEncoder


class T2ICoreSingleTextEncoder(BaseSingleTextEncoder):
    CHI_PROMPT = (
        'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:\n'
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.\n"
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.\n"
        "Here are examples of how to transform or refine prompts:\n"
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.\n"
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:\n"
        "User Prompt: "
    )

    def build_text_encoder(self):
        if self.id in [
            "sana-1.5/gemma-2-2b-it-300-bf16",
            "sana-sprint/gemma-2-2b-it-300-bf16",
        ]:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.encoder_dtype = torch.bfloat16
            self.text_max_length = 300
            self.require_state_type = "last_hidden_state"
            pretrained_model_name = "google/gemma-2-2b-it"
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
            self.tokenizer.padding_side = "right"
            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name, torch_dtype=self.encoder_dtype
            ).get_decoder()

        elif self.id in [
            "flux.1-dev/t5-512-bf16",
            "flux.1-krea-dev/t5-512-bf16",
            "flux.1-krea-dev/t5-256-bf16",
        ]:
            from transformers import T5EncoderModel, T5Tokenizer

            self.encoder_dtype = torch.bfloat16
            self.require_state_type = "last_hidden_state"
            self.use_mask = False
            if self.id == "flux.1-dev/t5-512-bf16":
                pipeline = "black-forest-labs/FLUX.1-dev"
                self.text_max_length = 512
            elif self.id == "flux.1-krea-dev/t5-512-bf16":
                pipeline = "black-forest-labs/FLUX.1-Krea-dev"
                self.text_max_length = 512
            elif self.id == "flux.1-krea-dev/t5-256-bf16":
                pipeline = "black-forest-labs/FLUX.1-Krea-dev"
                self.text_max_length = 256
            else:
                raise ValueError(f"Text encoder ID {self.id} is not supported")
            self.tokenizer = T5Tokenizer.from_pretrained(pipeline, subfolder="tokenizer_2")
            self.text_encoder = T5EncoderModel.from_pretrained(
                pipeline,
                subfolder="text_encoder_2",
                torch_dtype=self.encoder_dtype,
            )

        elif self.id in [
            "flux.1-dev/clip-77-bf16",
            "flux.1-krea-dev/clip-77-bf16",
        ]:
            from transformers import CLIPTextModel, CLIPTokenizer

            self.encoder_dtype = torch.bfloat16
            self.text_max_length = 77
            self.require_state_type = "pooler_output"
            self.use_mask = False
            if self.id == "flux.1-dev/clip-77-bf16":
                pipeline = "black-forest-labs/FLUX.1-dev"
            elif self.id == "flux.1-krea-dev/clip-77-bf16":
                pipeline = "black-forest-labs/FLUX.1-Krea-dev"
            else:
                raise ValueError(f"Text encoder ID {self.id} is not supported")
            self.tokenizer = CLIPTokenizer.from_pretrained(pipeline, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(
                pipeline,
                subfolder="text_encoder",
                torch_dtype=self.encoder_dtype,
            )

        elif self.id in [
            "z-image-turbo/qwen3-4b-512-bf16",
            "z-image/qwen3-4b-512-bf16",
        ]:
            from transformers import AutoModel, AutoTokenizer

            self.encoder_dtype = torch.bfloat16
            self.text_max_length = 512
            if self.id == "z-image-turbo/qwen3-4b-512-bf16":
                pipeline = "Tongyi-MAI/Z-Image-Turbo"
            elif self.id == "z-image/qwen3-4b-512-bf16":
                pipeline = "Tongyi-MAI/Z-Image"
            else:
                raise ValueError(f"Text encoder ID {self.id} is not supported")
            self.tokenizer = AutoTokenizer.from_pretrained(
                pipeline,
                subfolder="tokenizer",
                trust_remote_code=True,
            )
            self.text_encoder = AutoModel.from_pretrained(
                pipeline,
                subfolder="text_encoder",
                torch_dtype=self.encoder_dtype,
                trust_remote_code=True,
            )

        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    def convert_single_prompt(self, prompt: str):
        tokens = self.tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in self.tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def convert_prompt(self, prompts: list[str]):
        prompts = [self.convert_single_prompt(p) for p in prompts]
        return prompts

    @torch.no_grad()
    def get_text_embed_info(
        self,
        prompts: list[str],
        device: torch.device,
    ):

        if self.id in [
            "sana-1.5/gemma-2-2b-it-300-bf16",
            "sana-sprint/gemma-2-2b-it-300-bf16",
        ]:
            prompts_all = []
            for prompt in prompts:
                prompts_all.append(prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0].strip())

            chi_prompt = self.CHI_PROMPT
            prompts_all = [chi_prompt + prompt for prompt in prompts_all]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + self.text_max_length - 2

            tokens = self.tokenizer(
                prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            select_indices = [0] + list(range(-self.text_max_length + 1, 0))

            text_embeddings = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][
                :, :, select_indices
            ]

            text_embedding_masks = tokens.attention_mask[:, select_indices]

            return {
                "text_embeddings": text_embeddings,
                "text_embedding_masks": text_embedding_masks,
            }

        elif self.id in [
            "flux.1-dev/t5-512-bf16",
            "flux.1-krea-dev/t5-512-bf16",
            "flux.1-krea-dev/t5-256-bf16",
            "flux.1-dev/clip-77-bf16",
            "flux.1-krea-dev/clip-77-bf16",
        ]:
            prompts = self.convert_prompt(prompts)

            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.text_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if not self.use_mask:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
            else:
                text_input_masks = text_inputs.attention_mask
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=text_input_masks.to(device))

            if self.require_state_type == "last_hidden_state":
                prompt_embeds = prompt_embeds.last_hidden_state
            elif self.require_state_type == "pooler_output":
                prompt_embeds = prompt_embeds.pooler_output
            elif self.require_state_type == "text_embeds":
                prompt_embeds = prompt_embeds.text_embeds
            elif self.require_state_type == "hidden_states":
                prompt_embeds = prompt_embeds.hidden_states[1:]
                prompt_embeds = torch.stack(prompt_embeds, dim=0)
            else:
                raise NotImplementedError(f"State type {self.require_state_type} is not defined")

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            return {"text_embeddings": prompt_embeds}

        elif self.id in [
            "z-image-turbo/qwen3-4b-512-bf16",
            "z-image/qwen3-4b-512-bf16",
        ]:
            # ZImage uses chat template for text encoding
            formatted_prompts = []
            for prompt in prompts:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                formatted_prompts.append(formatted_prompt)

            text_inputs = self.tokenizer(
                formatted_prompts,
                padding="max_length",
                max_length=self.text_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(device)
            text_input_masks = text_inputs.attention_mask.to(device).bool()

            outputs = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_input_masks,
                output_hidden_states=True,
            )

            # ZImage uses hidden_states[-2] (second to last layer)
            prompt_embeds = outputs.hidden_states[-2]

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            return {"text_embeddings": prompt_embeds, "attention_mask": text_input_masks}

        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    @torch.no_grad()
    def get_null_embeddings(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        if self.id in [
            "z-image-turbo/qwen3-4b-512-bf16",
            "z-image/qwen3-4b-512-bf16",
        ]:
            # ZImage uses chat template for null embedding
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": ""}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            null_tokens = self.tokenizer(
                formatted_prompt,
                max_length=self.text_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)

            attention_mask = null_tokens.attention_mask.bool()
            outputs = self.text_encoder(
                input_ids=null_tokens.input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # ZImage uses hidden_states[-2]; filter out padding tokens
            embeddings = outputs.hidden_states[-2]
            valid = attention_mask[0]
            return embeddings[0][valid].unsqueeze(0)

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


class T2ICoreTextEncoder(BaseTextEncoder):
    def __init__(self, text_encoder_ids: Sequence[str]):
        super().__init__(text_encoder_ids)
        self.text_encoder_list: list[T2ICoreSingleTextEncoder]

    def build_single_text_encoder(self, text_encoder_id: str) -> T2ICoreSingleTextEncoder:
        return T2ICoreSingleTextEncoder(text_encoder_id)
