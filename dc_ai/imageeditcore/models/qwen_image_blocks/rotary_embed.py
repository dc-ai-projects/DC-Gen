# Rope for Qwen-Image-Edit was introduced by Qwen Team.
# The original implementation is by Alibaba Cloud, licensed under the Apache License 2.0. See https://github.com/QwenLM/Qwen-Image.

from typing import Tuple, Union

import torch


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

    return x_out.type_as(x)
