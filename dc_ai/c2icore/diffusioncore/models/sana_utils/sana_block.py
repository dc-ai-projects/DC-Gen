# SANA was introduced by Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, and Song Han in "SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers", see https://arxiv.org/abs/2410.10629.
# The original implementation is by NVIDIA CORPORATION & AFFILIATES, licensed under the Apache License 2.0. See https://github.com/NVlabs/Sana.

import torch
from torch import nn

from .....models.nn.ops import GLUMBConv, MBConv, ReLULinearAttention, SoftmaxAttention

__all__ = ["SanaClsTransformerBlock"]


class SanaClsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio,
        norm_eps: float = 1e-6,
        attention_bias: bool = True,
        ffn_mode="MBConv",
        use_linear_attn=False,
        attention_head_dim=32,
    ):
        super().__init__()
        self.hidden_dim = dim
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        if use_linear_attn:
            self.attn = ReLULinearAttention(
                in_channels=dim,
                out_channels=dim,
                dim=attention_head_dim,
                use_bias=(attention_bias, True),
                norm=None,
                eps=1e-8,
            )
        else:
            self.attn = SoftmaxAttention(
                in_channels=dim, out_channels=dim, dim=attention_head_dim, use_bias=(attention_bias, True), norm=None
            )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        if ffn_mode == "MBConv":
            self.ffn = MBConv(
                in_channels=dim,
                out_channels=dim,
                expand_ratio=mlp_ratio,
                use_bias=True,
                act_func=("silu", "silu", None),
                norm=None,
            )
        elif ffn_mode == "GLUMBConv":
            self.ffn = GLUMBConv(
                in_channels=dim,
                out_channels=dim,
                expand_ratio=mlp_ratio,
                use_bias=True,
                act_func=("silu", "silu", None),
                norm=None,
            )
        else:
            raise NotImplementedError(f"Feed Forward {ffn_mode} is not supported")
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(
        self,
        hidden_states,
        c,
        height,
        width,
    ) -> torch.Tensor:
        # 1. Modulation
        c = self.adaLN_modulation(c)  # bsz * (6*dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = c.chunk(6, dim=1)  # bsz * 1 * dim

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)  # bsz * (H*W) * dim
        norm_hidden_states = norm_hidden_states * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)

        attn_output = self.attn(norm_hidden_states)
        attn_output = attn_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        # 3. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)

        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        ffn_output = self.ffn(norm_hidden_states)
        ffn_output = ffn_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ffn_output

        return hidden_states
