# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/norms.py.

import torch
import torch.nn as nn


class SanaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, scale_factor=1.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim) * scale_factor)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self._norm(x.float())).type_as(x)
