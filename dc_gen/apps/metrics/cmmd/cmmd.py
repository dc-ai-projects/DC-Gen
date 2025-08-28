# CMMD was introduced by Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar in "Rethinking FID: Towards a Better Evaluation Metric for Image Generation", see https://arxiv.org/abs/2401.09603.
# The original implementation is by the Google Research Authors, licensed under the Apache License 2.0. See https://github.com/google-research/google-research/tree/master/cmmd.
# The PyTorch version is adapted by https://github.com/sayakpaul/cmmd-pytorch, licensed under the Apache License 2.0.

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as transforms

from ...utils.dist import dist_init, is_dist_initialized, is_master, sync_tensor
from .model import ClipEmbeddingModel


def mmd_part(
    x: torch.Tensor,
    y: torch.Tensor,
    x_sum_square: torch.Tensor,
    y_sum_square: torch.Tensor,
    gamma: float,
    chunk_size: int = 10000,
):
    assert x.shape[0] == y.shape[0] == x_sum_square.shape[0] == y_sum_square.shape[0]
    num_samples = x.shape[0]
    k_list = []
    for start in range(0, num_samples, chunk_size):
        x_chunk = x[start : min(start + chunk_size, num_samples)]
        x_sum_square_chunk = x_sum_square[start : min(start + chunk_size, num_samples)]
        k_list.append(
            (-gamma * (-2 * (x_chunk @ y.T) + x_sum_square_chunk.unsqueeze(1) + y_sum_square.unsqueeze(0)))
            .exp()
            .mean(1)
        )
    return torch.cat(k_list).mean()


def mmd(x: np.ndarray, y: np.ndarray, sigma: float = 10, scale: float = 1000) -> torch.Tensor:
    """Memory-efficient MMD implementation in PyTorch.

    This implements the minimum-variance/biased version of the estimator described
    in Eq.(5) of
    https://jmlr.csail.mit.edu/papers/volume13/gretton12a/gretton12a.pdf.
    As described in Lemma 6's proof in that paper, the unbiased estimate and the
    minimum-variance estimate for MMD are almost identical.

    Args:
    x: The first set of embeddings of shape (n, embedding_dim).
    y: The second set of embeddings of shape (n, embedding_dim).
    sigma: The bandwidth parameter for the Gaussian RBF kernel. See the paper for more details.
    scale: The following is used to make the metric more human readable. See the paper for more details.

    Returns:
    The MMD distance between x and y embedding sets.
    """
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()

    x_sum_square = (x**2).sum(axis=1)  # (n,)
    y_sum_square = (y**2).sum(axis=1)  # (n,)

    gamma = 1 / (2 * sigma**2)
    k_xx = mmd_part(x, x, x_sum_square, x_sum_square, gamma)
    k_xy = mmd_part(x, y, x_sum_square, y_sum_square, gamma)
    k_yy = mmd_part(y, y, y_sum_square, y_sum_square, gamma)

    return scale * (k_xx + k_yy - 2 * k_xy)


@dataclass
class CMMDStatsConfig:
    save_path: Optional[str] = None
    ref_path: Optional[str] = None


class CMMDStats:
    def __init__(self, cfg: CMMDStatsConfig):
        self.cfg = cfg
        if cfg.ref_path is not None:
            assert os.path.exists(cfg.ref_path)
        # embedding model
        self.model = ClipEmbeddingModel()
        self.embs = []

    @torch.no_grad()
    def add_data(self, batch: torch.Tensor):
        # (B, 3, H, W)
        if batch.dtype == torch.uint8:
            batch = batch / 255
        else:
            # to simulate storing and loading generated images
            # reference: torchvision save_image
            # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
            batch_quantized = (255 * batch + 0.5).clamp(0, 255).to(torch.uint8)
            batch = batch_quantized / 255
        batch: np.ndarray = batch.cpu().numpy().transpose(0, 2, 3, 1)  # (B, 3, H, W) -> (B, H, W, 3)
        embs = self.model.embed(batch).cpu().numpy()
        self.embs.append(embs)

    def get_stats(self):
        embs = np.concatenate(self.embs, axis=0)
        # distributed
        if is_dist_initialized():
            embs = sync_tensor(torch.from_numpy(embs).cuda(), reduce="cat").cpu().numpy()

        if is_master() and self.cfg.save_path is not None:
            os.makedirs(os.path.dirname(self.cfg.save_path), exist_ok=True)
            np.save(self.cfg.save_path, embs)

        return embs

    def compute_cmmd(self):
        embs1 = self.get_stats()  # every node must enter get_stats
        if not is_master():
            return 0
        embs2 = np.load(self.cfg.ref_path)
        if embs1.shape[0] >= embs2.shape[0]:
            embs1 = embs1[: embs2.shape[0]]
        else:
            if is_master():
                print(
                    f"warning: generated samples {embs1.shape[0]} should be larger than or equal to reference samples {embs2.shape[0]} for cmmd"
                )
            embs2 = embs2[: embs1.shape[0]]
        cmmd = mmd(embs1, embs2).item()
        return cmmd

    def reset(self):
        self.embs = []
