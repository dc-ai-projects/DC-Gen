# Modified from ``https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/model.py''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import torch
from torch import nn


def flash_attention(
    q,
    k,
    v,
    k_lens=None,
    dropout_p=0.0,
    window_size=(-1, -1),
    dtype=torch.bfloat16,
):
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == "cuda" and q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q = half(q.flatten(0, 1))
    k = half(k.flatten(0, 1))
    v = half(v.flatten(0, 1))

    q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    import flash_attn

    x = flash_attn.flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens])
        .cumsum(0, dtype=torch.int32)
        .to(q.device, non_blocking=True),
        max_seqlen_q=lq,
        max_seqlen_k=lk,
        dropout_p=dropout_p,
        softmax_scale=None,
        causal=False,
        window_size=window_size,
        deterministic=False,
    ).unflatten(0, (b, lq))

    return x.type(out_dtype)


def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        output.append(x_i)
    return torch.stack(output).float()


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        from .norm import WanRMSNorm

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        x_dtype = x.dtype

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        x = flash_attention(q, k, v, k_lens=seq_lens, window_size=self.window_size)

        x = x.flatten(2).to(x_dtype)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=None)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        eps=1e-6,
        text_max_length=512,
    ):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)

        from .norm import WanRMSNorm

        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.text_max_length = text_max_length

    def forward(self, x, context):
        image_context_length = context.shape[1] - self.text_max_length
        context_img = context[:, :image_context_length]
        context = context[:, image_context_length:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)

        x = flash_attention(q, k, v, k_lens=None)

        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x
