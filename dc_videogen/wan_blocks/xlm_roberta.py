# Modified from transformers.models.xlm_roberta.modeling_xlm_roberta
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import torch.nn as nn

__all__ = ["XLMRoberta"]


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-5):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, post_norm, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.eps = eps

        self.attn = SelfAttention(dim, num_heads, eps)
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim, eps=eps)


class XLMRoberta(nn.Module):
    def __init__(
        self,
        vocab_size=250002,
        max_seq_len=514,
        type_size=1,
        pad_id=1,
        dim=1024,
        num_heads=16,
        num_layers=24,
        post_norm=True,
        eps=1e-5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.post_norm = post_norm
        self.eps = eps

        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.type_embedding = nn.Embedding(type_size, dim)
        self.pos_embedding = nn.Embedding(max_seq_len, dim, padding_idx=pad_id)

        self.blocks = nn.ModuleList([AttentionBlock(dim, num_heads, post_norm, eps) for _ in range(num_layers)])

        # norm layer
        self.norm = nn.LayerNorm(dim, eps=eps)
