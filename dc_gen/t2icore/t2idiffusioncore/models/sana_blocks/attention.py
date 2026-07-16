# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/sana_blocks.py.

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention as Attention_

from .norm import SanaRMSNorm


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        if qk_norm:
            self.q_norm = SanaRMSNorm(d_model, scale_factor=1.0, eps=1e-6)
            self.k_norm = SanaRMSNorm(d_model, scale_factor=1.0, eps=1e-6)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape
        first_dim = B

        q = self.q_linear(x)
        kv = self.kv_linear(cond).view(first_dim, -1, 2, C)
        k, v = kv.unbind(2)
        q = self.q_norm(q).view(first_dim, -1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(first_dim, -1, self.num_heads, self.head_dim)
        v = v.view(first_dim, -1, self.num_heads, self.head_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None and mask.ndim == 2:
            mask = (1 - mask.to(x.dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LiteLA(Attention_):
    PAD_VAL = 1

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim=32,
        eps=1e-15,
        use_bias=False,
        qk_norm=False,
        norm_eps=1e-5,
    ):
        heads = heads or int(out_dim // dim * heads_ratio)
        super().__init__(in_dim, num_heads=heads, qkv_bias=use_bias)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dim = out_dim // heads  # TODO: need some change
        self.eps = eps

        self.kernel_func = nn.ReLU(inplace=False)
        if qk_norm:
            self.q_norm = SanaRMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
            self.k_norm = SanaRMSNorm(in_dim, scale_factor=1.0, eps=norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=LiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(self, x: torch.Tensor, mask=None, HW=None, block_id=None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(2)  # B, N, 3, C --> B, N, C
        dtype = q.dtype

        q = self.q_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.k_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)

        q = q.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)
        k = k.reshape(B, C // self.dim, self.dim, N).transpose(-1, -2)  # (B, h, N, h_d)
        v = v.reshape(B, C // self.dim, self.dim, N)  # (B, h, h_d, N)

        out = self.attn_matmul(q, k, v).to(dtype)

        out = out.view(B, C, N).permute(0, 2, 1)  # B, N, C
        out = self.proj(out)

        if not self.training:
            out = out.clip(-65504, 65504)

        return out

    @property
    def module_str(self) -> str:
        _str = type(self).__name__ + "("
        eps = f"{self.eps:.1E}"
        _str += f"i={self.in_dim},o={self.out_dim},h={self.heads},d={self.dim},eps={eps}"
        return _str

    def __repr__(self):
        return f"EPS{self.eps}-" + super().__repr__()
