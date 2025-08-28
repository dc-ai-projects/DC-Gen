# Precision and Recall was introduced by Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila in "Improved Precision and Recall Metric for Assessing Generative Models", see https://arxiv.org/abs/1904.06991.
# The original implementation is by NVIDIA Corporation, licensed under the Creative Commons BY-NC 4.0. See https://github.com/kynkaat/improved-precision-and-recall-metric.
# The PyTorch version is adapted by OpenAI, licensed under the MIT license. See https://github.com/openai/guided-diffusion.

import numpy as np
import torch


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_size=3,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param eps: small number for numerical stability.
        """
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_size = nhood_size
        self.eps = eps

    def manifold_radii(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (num_images, 2048)
        """
        num_images = len(features)
        device = features.device

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = torch.zeros(num_images, dtype=torch.float32, device=device)
        features_sum_square = (features**2).sum(dim=1)  # (num_images)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]
            distance_batch = (
                (row_batch**2).sum(dim=1)[:, None] + features_sum_square[None, :] - 2 * row_batch @ features.T
            )  # (row, all)
            radii[begin1:end1] = torch.topk(distance_batch, self.nhood_size + 1, dim=1, largest=False).values[:, -1]

        return radii

    def evaluate_pr(self, features_1: torch.Tensor, features_2: torch.Tensor) -> tuple[float, float]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :return: a tuple of arrays for (precision, recall):
                 - precision: a torch.Tensor of length K1
                 - recall: a torch.Tensor of length K2
        """
        radii_1 = self.manifold_radii(features_1)
        radii_2 = self.manifold_radii(features_2)

        device = features_1.device
        features_1_status = torch.zeros(len(features_1), dtype=bool, device=device)
        features_2_status = torch.zeros(len(features_2), dtype=bool, device=device)

        features_1_sum_square = (features_1**2).sum(dim=1)  # (num_images)
        features_2_sum_square = (features_2**2).sum(dim=1)  # (num_images)

        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                distance = (
                    features_1_sum_square[begin_1:end_1, None]
                    + features_2_sum_square[None, begin_2:end_2]
                    - 2 * batch_1 @ batch_2.T
                )  # (row, col)
                batch_1_in = torch.any(distance <= radii_2[None, begin_2:end_2], dim=1)
                batch_2_in = torch.any(distance <= radii_1[begin_1:end_1, None], dim=0)
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return features_2_status.float().mean().item(), features_1_status.float().mean().item()
