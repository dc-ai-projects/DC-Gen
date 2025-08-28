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
import torch.nn.functional as F

from ...utils.dist import is_dist_initialized, is_master, sync_tensor


@dataclass
class SelfKnnStatsConfig:
    """
    Using same dataset as Knn train dataset and Knn eval dataset.
    """

    nearest_neighbors: tuple[int, ...] = (1, 5, 10, 20)
    temperature: float = 0.07


class SelfKnnStats:
    def __init__(self, cfg: SelfKnnStatsConfig):
        self.cfg = cfg

        self.features_rank: list[torch.Tensor] = []
        self.labels_rank: list[torch.Tensor] = []

    @torch.inference_mode()
    def add_data(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self.features_rank.append(features.cpu())
        self.labels_rank.append(labels.cpu())

    @torch.inference_mode()
    def get_stats(self):
        features = torch.cat(self.features_rank, dim=0).cuda()
        labels = torch.cat(self.labels_rank, dim=0).cuda()

        if is_dist_initialized():
            features = sync_tensor(features, reduce="cat")
            labels = sync_tensor(labels, reduce="cat")
        features = F.normalize(features.float(), dim=-1)
        return features, labels

    @torch.inference_mode()
    def compute(self) -> dict[str, float]:
        features, labels = self.get_stats()
        if not is_master():
            return {}

        similarity = features @ features.T
        # fill diagonal with -inf to exclude self-comparison
        similarity.fill_diagonal_(-float("inf"))
        max_k = max(self.cfg.nearest_neighbors)
        topk_values, topk_indices = similarity.topk(max_k, dim=1, largest=True)
        topk_labels = labels[topk_indices]
        results = {}
        N = features.shape[0]
        for k in self.cfg.nearest_neighbors:
            weights = F.softmax(topk_values[:, :k] / self.cfg.temperature, dim=1)

            one_hot = F.one_hot(topk_labels[:, :k]).float()

            # [N, 1, k] @ [N, k, C] -> [N, 1, C] -> [N, C]
            class_probs = (weights.unsqueeze(1) @ one_hot).squeeze(1)

            pred_labels = class_probs.argmax(dim=1)
            correct = (pred_labels == labels).sum().item()
            accuracy = correct / N

            results[f"self_knn_top{k}_acc"] = accuracy

        return results
