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

from dataclasses import dataclass

import torch

from ..utils.dist import sync_tensor


@dataclass
class CustomLPIPSStatsConfig:
    net: str = "vgg"


class CustomLPIPSStats:
    def __init__(self, cfg: CustomLPIPSStatsConfig, device: torch.device):
        from lpips import LPIPS

        self.cfg = cfg
        self.model = LPIPS(net=cfg.net).to(device).eval()
        self.lpips_sum, self.lpips_cnt = 0, 0

    @torch.no_grad()
    def add_data(self, image_ref: torch.Tensor, image_pred: torch.Tensor):
        """
        value range: [0, 1]
        """
        lp = self.model(image_ref.float(), image_pred.float(), normalize=True)
        self.lpips_sum += lp.sum().item()
        self.lpips_cnt += image_ref.shape[0]

    def compute(self):
        lpips_sum = sync_tensor(self.lpips_sum, reduce="sum")
        lpips_cnt = sync_tensor(self.lpips_cnt, reduce="sum")

        if isinstance(lpips_sum, torch.Tensor):
            lpips_sum = lpips_sum.item()
        if isinstance(lpips_cnt, torch.Tensor):
            lpips_cnt = lpips_cnt.item()

        result = {"custom_lpips": lpips_sum / lpips_cnt}
        return result

    def reset(self):
        self.lpips_sum, self.lpips_cnt = 0, 0
