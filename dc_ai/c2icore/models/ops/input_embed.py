# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["AdaptivePatchEmbed"]


class AdaptivePatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        bias: bool,
        share_weights: bool,
        kernel_size: Optional[int] = None,
    ):
        super().__init__()
        self.share_weights = share_weights
        self.patch_size = (patch_size, patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        if kernel_size is None:
            kernel_size = patch_size
            padding = 0
        else:
            padding = kernel_size // 2
        if share_weights:
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size, padding=padding, bias=bias
            )
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor, keep_2d: bool = False) -> torch.Tensor:
        # (B, C, H, W)
        in_channels = x.shape[1]
        if self.share_weights:
            x = F.conv2d(
                x,
                self.proj.weight[:, :in_channels],
                self.proj.bias,
                self.proj.stride,
                self.proj.padding,
                self.proj.dilation,
                self.proj.groups,
            )
        else:
            raise NotImplementedError
        if not keep_2d:
            x = x.flatten(2).transpose(1, 2)
        return x
