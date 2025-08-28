# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/builder.py and https://github.com/NVlabs/Sana/blob/main/app/sana_pipeline.py.

import re
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import MISSING

from ..models.utils.network import get_dtype_from_str


@dataclass
class SingleTextEncoderConfig:
    # basic
    name: str = MISSING
    text_max_length: int = 300
    encoder_dtype: str = "bf16"

    # init
    pipeline: Optional[str] = None
    subfolder: Optional[str] = None

    # encode
    require_state_type: str = "last_hidden_state"
    use_mask: bool = False


class SingleTextEncoder(nn.Module):
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

    def __init__(self, cfg: SingleTextEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder_dtype = get_dtype_from_str(cfg.encoder_dtype)

        if cfg.name in [
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/gemma-2-2b",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2-0.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
        ]:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(cfg.name)
            self.tokenizer.padding_side = "right"
            self.text_encoder = AutoModelForCausalLM.from_pretrained(
                cfg.name, torch_dtype=self.encoder_dtype
            ).get_decoder()

        elif cfg.name in [
            "DeepFloyd/t5-v1_1-xxl",
            "google/t5-v1_1-small",
            "google/t5-v1_1-base",
            "google/t5-v1_1-large",
            "google/t5-v1_1-xl",
            "google/t5-v1_1-xxl",
        ]:
            from transformers import T5EncoderModel, T5Tokenizer

            self.tokenizer = T5Tokenizer.from_pretrained(cfg.name)
            self.text_encoder = T5EncoderModel.from_pretrained(cfg.name, torch_dtype=self.encoder_dtype)

        elif cfg.name in [
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-proj",
            "openai/clip-vit-base-patch32-proj",
        ]:
            from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

            if cfg.pipeline is not None:
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    cfg.pipeline,
                    subfolder=cfg.subfolder.replace("text_encoder", "tokenizer"),
                )
                if "proj" in cfg.name:
                    self.text_encoder = CLIPTextModelWithProjection.from_pretrained(
                        cfg.pipeline,
                        subfolder=cfg.subfolder,
                        torch_dtype=self.encoder_dtype,
                    )
                else:
                    self.text_encoder = CLIPTextModel.from_pretrained(
                        cfg.pipeline,
                        subfolder=cfg.subfolder,
                        torch_dtype=self.encoder_dtype,
                    )
            else:
                self.tokenizer = CLIPTokenizer.from_pretrained(cfg.name)
                self.text_encoder = CLIPTextModel.from_pretrained(
                    cfg.name,
                    torch_dtype=self.encoder_dtype,
                )

        elif cfg.name in [
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ]:
            from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.text_encoder = LlamaForCausalLM.from_pretrained(
                cfg.name,
                output_hidden_states=True,
                output_attentions=True,
                torch_dtype=self.encoder_dtype,
            )

        else:
            raise ValueError(f"Text encoder {cfg.name} is not supported")

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
        if self.cfg.name in [
            "google/gemma-2b",
            "google/gemma-2b-it",
            "google/gemma-2-2b",
            "google/gemma-2-2b-it",
            "google/gemma-2-9b",
            "google/gemma-2-9b-it",
            "Qwen/Qwen2-0.5B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
        ]:
            prompts_all = []
            for prompt in prompts:
                prompts_all.append(prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0].strip())

            chi_prompt = self.CHI_PROMPT
            prompts_all = [chi_prompt + prompt for prompt in prompts_all]
            num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_prompt_tokens + self.cfg.text_max_length - 2

            tokens = self.tokenizer(
                prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            select_indices = [0] + list(range(-self.cfg.text_max_length + 1, 0))

            text_embeddings = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][
                :, :, select_indices
            ]

            text_embedding_masks = tokens.attention_mask[:, select_indices]

            return {
                "text_embeddings": text_embeddings,
                "text_embedding_masks": text_embedding_masks,
            }

        elif self.cfg.name in [
            "DeepFloyd/t5-v1_1-xxl",
            "google/t5-v1_1-small",
            "google/t5-v1_1-base",
            "google/t5-v1_1-large",
            "google/t5-v1_1-xl",
            "google/t5-v1_1-xxl",
            "openai/clip-vit-large-patch14",
            "openai/clip-vit-large-patch14-proj",
            "openai/clip-vit-base-patch32-proj",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        ]:
            prompts = self.convert_prompt(prompts)

            text_inputs = self.tokenizer(
                prompts,
                padding="max_length",
                max_length=self.cfg.text_max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids

            if not self.cfg.use_mask:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
            else:
                text_input_masks = text_inputs.attention_mask
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=text_input_masks.to(device))

            if self.cfg.require_state_type == "last_hidden_state":
                prompt_embeds = prompt_embeds.last_hidden_state
            elif self.cfg.require_state_type == "pooler_output":
                prompt_embeds = prompt_embeds.pooler_output
            elif self.cfg.require_state_type == "text_embeds":
                prompt_embeds = prompt_embeds.text_embeds
            elif self.cfg.require_state_type == "hidden_states":
                prompt_embeds = prompt_embeds.hidden_states[1:]
                prompt_embeds = torch.stack(prompt_embeds, dim=0)
            else:
                raise NotImplementedError(f"State type {self.cfg.require_state_type} is not defined")

            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
            return {"text_embeddings": prompt_embeds}

    @torch.no_grad()
    def get_null_embeddings(
        self,
        device: torch.device,
        require_state_type: Optional[str] = None,
    ) -> torch.Tensor:
        null_tokens = self.tokenizer(
            "",
            max_length=self.cfg.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        null_embeddings = self.text_encoder(null_tokens.input_ids, null_tokens.attention_mask)
        if require_state_type is None:
            null_text_embeddings = null_embeddings[0]
        elif require_state_type == "hidden_states":
            null_text_embeddings = null_embeddings.hidden_states[1:]
            null_text_embeddings = torch.stack(null_text_embeddings, dim=0)
        else:
            raise NotImplementedError(f"State type {require_state_type} is not defined")

        return null_text_embeddings


class TextEncoder(nn.Module):
    def __init__(self, cfg: dict[str, SingleTextEncoderConfig]):
        super().__init__()
        self.cfg = cfg
        text_encoder_list: list[SingleTextEncoder] = []
        for _, text_encoder_cfg in cfg.items():
            text_encoder = SingleTextEncoder(text_encoder_cfg)
            text_encoder_list.append(text_encoder)
        self.text_encoder_list: list[SingleTextEncoder] = nn.ModuleList(text_encoder_list)

    @torch.no_grad()
    def get_text_embed_info(self, prompts: list[str], device: torch.device):
        text_embed_info = {}
        for text_encoder in self.text_encoder_list:
            text_embed_info[text_encoder.cfg.name] = text_encoder.get_text_embed_info(prompts, device)
        return text_embed_info
