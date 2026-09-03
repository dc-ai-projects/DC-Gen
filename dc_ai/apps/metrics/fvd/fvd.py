# FVD was introduced by Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly in "Towards Accurate Generative Models of Video: A New Metric & Challenges", see https://arxiv.org/abs/1812.01717.
# The original implementation is by The Google Research Authors, licensed under the Apache License 2.0. See https://github.com/google-research/google-research/tree/master/frechet_video_distance.
# The PyTorch version is from https://github.com/JunyaoHu/common_metrics_on_video_quality.

import itertools
import math
import os
from dataclasses import dataclass
from typing import Optional

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import sqrtm

from ...utils.dist import dist_barrier, dist_init, is_dist_initialized, is_master, sync_tensor

__all__ = ["FVDStatsConfig", "FVDStats"]


def load_i3d_pretrained(device=torch.device("cpu")):
    i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
    filepath = os.path.join("assets/checkpoints/pretrained", "i3d_torchscript.pt")
    if not os.path.exists(filepath):
        print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
        if is_master():
            os.system(f"wget {i3D_WEIGHTS_URL} -O {filepath}")
    dist_barrier()
    i3d = torch.jit.load(filepath).eval().to(device)
    return i3d


def preprocess_single_video(video, resolution=224):
    # video: CTHW, [0, 1]
    _, _, h, w = video.shape

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode="bilinear", align_corners=False)

    # center crop
    _, _, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start : h_start + resolution, w_start : w_start + resolution]

    # [0, 1] -> [-1, 1]
    video = (video - 0.5) * 2

    return video.contiguous()


def frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real) -> float:
    m = np.square(mu_gen - mu_real).sum()
    s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)


@dataclass
class FVDStatsConfig:
    save_path: Optional[str] = None
    ref_path: Optional[str] = None


class FVDStats:
    def __init__(self, cfg: FVDStatsConfig):
        self.cfg = cfg
        # inception model
        self.device = torch.device("cuda")
        self.model = load_i3d_pretrained(self.device)

        # value should be floats within [0, 1], on gpu
        self.num_samples = 0
        self.pred_sum = None
        self.pred_dot_product_sum = None

    @torch.no_grad()
    def add_data(self, batch: torch.Tensor):
        # batch: BCTHW [0, 1]
        if batch.dtype == torch.uint8:
            batch = batch / 255
        else:
            # to simulate storing and loading generated images
            # reference: torchvision save_image
            # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
            batch_quantized = (255 * batch + 0.5).clamp(0, 255).to(torch.uint8)
            batch = batch_quantized / 255

        batch = torch.stack([preprocess_single_video(video) for video in batch]).to(self.device)

        pred = self.model(batch, rescale=False, resize=False, return_features=True).detach().cpu().numpy()

        self.num_samples += pred.shape[0]
        if self.pred_sum is None:
            self.pred_sum = np.zeros(pred.shape[1])
        if self.pred_dot_product_sum is None:
            self.pred_dot_product_sum = np.zeros((pred.shape[1], pred.shape[1]))
        self.pred_sum += pred.sum(axis=0)
        self.pred_dot_product_sum += pred.T @ pred

    def get_stats(self):
        num_samples, pred_sum, pred_dot_product_sum = self.num_samples, self.pred_sum, self.pred_dot_product_sum

        if is_dist_initialized():
            num_samples = sync_tensor(torch.tensor(num_samples).cuda(), reduce="sum").cpu().numpy()
            pred_sum = sync_tensor(torch.from_numpy(pred_sum).cuda(), reduce="sum").cpu().numpy()
            pred_dot_product_sum = (
                sync_tensor(torch.from_numpy(pred_dot_product_sum).cuda(), reduce="sum").cpu().numpy()
            )
            if not is_master():
                return None, None
        mu = pred_sum / num_samples
        sigma = (pred_dot_product_sum - num_samples * (mu[:, None] @ mu[None])) / (num_samples - 1)

        if self.cfg.save_path is not None:
            os.makedirs(os.path.dirname(self.cfg.save_path), exist_ok=True)
            np.savez(self.cfg.save_path, mu=mu, sigma=sigma)
        return mu, sigma

    def compute_fvd(
        self, ref_path: Optional[str] = None, mu2: Optional[np.ndarray] = None, sigma2: Optional[np.ndarray] = None
    ):
        mu1, sigma1 = self.get_stats()  # every node must enter get_stats

        # only compute fvd score at master
        if not is_master():
            return 0

        if mu2 is None or sigma2 is None:
            if ref_path is None:
                ref_path = self.cfg.ref_path
            ref_data = np.load(ref_path)
            mu2, sigma2 = ref_data["mu"], ref_data["sigma"]
        fvd = frechet_distance(mu1, sigma1, mu2, sigma2)
        return fvd

    def reset(self):
        self.num_samples = 0
        self.pred_sum = None
        self.pred_dot_product_sum = None
