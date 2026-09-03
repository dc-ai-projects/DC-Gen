import os
from typing import Optional

import torch
from torch import nn


# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana_blocks.py.
class CaptionEmbedder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        uncond_prob: float,
        act_layer: type[nn.Module] = lambda: nn.GELU(approximate="tanh"),
        token_num: int = 120,
        text_encoder_id: str = "sana-1.5/gemma-2-2b-it-300-bf16",
    ):
        super().__init__()
        from timm.models.vision_transformer import Mlp

        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )

        null_embedding_path = os.path.join("assets/data/null_text_embeddings", f"{text_encoder_id}.pth")
        null_embedding = torch.load(null_embedding_path, weights_only=True)
        self.register_buffer("y_embedding", null_embedding)

        self.uncond_prob = uncond_prob

    def token_drop(self, caption: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(
        self, caption: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ):
        if self.training:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0

        if (self.training and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)

        caption = self.y_proj(caption)

        return caption


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
