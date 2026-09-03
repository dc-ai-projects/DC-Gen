# RMS Norm was introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = hidden_states * self.weight

        return hidden_states
