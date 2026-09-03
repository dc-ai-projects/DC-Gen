from collections.abc import Sequence

import torch
import torch.nn as nn


class BaseSingleTextEncoder(nn.Module):
    def __init__(self, text_encoder_id: str):
        super().__init__()
        self.id = text_encoder_id

        self.build_text_encoder()

    def build_text_encoder(self):
        raise NotImplementedError

    @torch.no_grad()
    def get_text_embed_info(
        self,
        prompts: list[str],
        device: torch.device,
    ):
        raise NotImplementedError

    @torch.no_grad()
    def get_null_embeddings(
        self,
        device: torch.device,
    ) -> torch.Tensor:
        raise NotImplementedError


class BaseTextEncoder(nn.Module):
    def __init__(self, text_encoder_ids: Sequence[str]):
        super().__init__()
        if isinstance(text_encoder_ids, str):
            raise TypeError("text_encoder_ids must be a sequence of complete IDs, not a single string")
        self.ids = tuple(text_encoder_ids)
        text_encoder_list: list[BaseSingleTextEncoder] = []
        unique_text_encoder_ids: set[str] = set()
        for text_encoder_id in self.ids:
            if text_encoder_id in unique_text_encoder_ids:
                raise ValueError(f"Duplicate text encoder ID {text_encoder_id}")
            unique_text_encoder_ids.add(text_encoder_id)
            text_encoder = self.build_single_text_encoder(text_encoder_id)
            text_encoder_list.append(text_encoder)
        self.text_encoder_list: list[BaseSingleTextEncoder] = nn.ModuleList(text_encoder_list)

    def build_single_text_encoder(self, text_encoder_id: str) -> BaseSingleTextEncoder:
        raise NotImplementedError

    @torch.no_grad()
    def get_text_embed_info(self, prompts: list[str], device: torch.device):
        text_embed_info = {}
        for text_encoder in self.text_encoder_list:
            text_embed_info[text_encoder.id] = text_encoder.get_text_embed_info(prompts, device)
        return text_embed_info

    # FSDP interface
    @torch.no_grad()
    def forward(self, prompts: list[str], device: torch.device):
        return self.get_text_embed_info(prompts, device)
