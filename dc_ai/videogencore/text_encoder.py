# Wan-2.1 was introduced by Wan Team in "Wan: Open and Advanced Large-Scale Video Generative Models", see https://arxiv.org/abs/2503.20314.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0. See https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/t5.py.

from collections.abc import Sequence

import torch

from ..apps.text_encoder import BaseSingleTextEncoder, BaseTextEncoder


class VideoGenCoreSingleTextEncoder(BaseSingleTextEncoder):
    WAN_NEGATIVE_PROMPT = "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走"
    WAN_I2V_NEGATIVE_PROMPT = "镜头晃动," + WAN_NEGATIVE_PROMPT

    def build_text_encoder(self):
        if self.id in [
            "wan2.1-t2v/umt5-512-bf16",
            "wan2.1-i2v/umt5-512-bf16",
        ]:
            from transformers import T5Tokenizer

            from .models.wan_blocks.t5 import T5Encoder

            self.encoder_dtype = torch.bfloat16
            self.text_max_length = 512
            pretrained_path = "assets/checkpoints/t2v/models_t5_umt5-xxl-enc-bf16.pth"
            assert pretrained_path is not None, "Must load from pretrained model for WanVideo."

            self.tokenizer = T5Tokenizer.from_pretrained("google/umt5-xxl")
            self.text_encoder = T5Encoder()
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            self.text_encoder.load_state_dict(checkpoint)
            self.text_encoder = self.text_encoder.to(self.encoder_dtype).eval()
        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    @torch.no_grad()
    def get_text_embed_info(
        self,
        prompts: list[str],
        device: torch.device,
    ):
        if self.id in [
            "wan2.1-t2v/umt5-512-bf16",
            "wan2.1-i2v/umt5-512-bf16",
        ]:
            bs = len(prompts)
            tokens = self.tokenizer(
                prompts,
                max_length=self.text_max_length,
                return_tensors="pt",
                add_special_tokens=True,
                padding=True,
            ).to(device)

            attention_mask = tokens.attention_mask[:, : self.text_max_length]  # truncate overlong sequences
            input_ids = tokens.input_ids[:, : self.text_max_length]

            seq_lens = attention_mask.gt(0).sum(dim=1).long()
            pad_length = self.text_max_length - seq_lens.max().item()
            input_ids_pad = torch.zeros(bs, pad_length).to(device=device, dtype=input_ids.dtype)

            input_ids = torch.cat((input_ids, input_ids_pad), dim=1)
            attention_mask_pad = torch.zeros(bs, pad_length).to(device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, attention_mask_pad), dim=1)

            text_embeddings = self.text_encoder(input_ids, attention_mask)

            text_embeddings = torch.stack(
                [
                    torch.cat([u[:v], u.new_zeros(self.text_max_length - v, u.size(1))], dim=0)
                    for u, v in zip(text_embeddings, seq_lens)
                ]
            )

            return {
                "text_embeddings": text_embeddings,
            }
        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")

    @torch.no_grad()
    def get_null_embeddings(
        self,
        device: torch.device,
    ):
        if self.id == "wan2.1-t2v/umt5-512-bf16":
            prompts = [self.WAN_NEGATIVE_PROMPT]
            return self.get_text_embed_info(prompts, device)["text_embeddings"][0]
        elif self.id == "wan2.1-i2v/umt5-512-bf16":
            prompts = [self.WAN_I2V_NEGATIVE_PROMPT]
            return self.get_text_embed_info(prompts, device)["text_embeddings"][0]
        else:
            raise ValueError(f"Text encoder ID {self.id} is not supported")


class VideoGenCoreTextEncoder(BaseTextEncoder):
    def __init__(self, text_encoder_ids: Sequence[str]):
        super().__init__(text_encoder_ids)
        self.text_encoder_list: list[VideoGenCoreSingleTextEncoder]

    def build_single_text_encoder(self, text_encoder_id: str) -> VideoGenCoreSingleTextEncoder:
        return VideoGenCoreSingleTextEncoder(text_encoder_id)
