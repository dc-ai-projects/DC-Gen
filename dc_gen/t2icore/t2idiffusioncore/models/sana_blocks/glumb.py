# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana/blob/main/diffusion/model/nets/basic_modules.py.

from typing import Optional

import torch

from .....models.nn.ops import GLUMBConv


class SanaGLUMBConv(GLUMBConv):
    def forward(self, x: torch.Tensor, HW: Optional[tuple[int, int]] = None) -> torch.Tensor:
        B, N, C = x.shape
        if HW is None:
            H = W = int(N**0.5)
        else:
            H, W = HW

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = super().forward(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x
