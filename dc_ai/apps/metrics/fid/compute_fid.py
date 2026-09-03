# FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see https://arxiv.org/abs/1706.08500.
# The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0. See https://github.com/bioinf-jku/TTUR.
# The PyTorch version is adapted by https://github.com/mseitzer/pytorch-fid, licensed under the Apache License 2.0.

from dataclasses import dataclass, field
from typing import Optional

import torch
import torchvision.transforms as transforms
from omegaconf import MISSING, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...data_provider.sampler import DistributedRangedSampler
from ...utils.dist import dist_init, get_dist_local_rank, get_dist_rank, get_dist_size, is_master
from ...utils.image import DMCrop, ImageDataset
from .fid import FIDStats, FIDStatsConfig


@dataclass
class ComputeFIDConfig:
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    data_dir: str = MISSING
    ref_data_dir: Optional[str] = None
    resolution: int = 256
    suffix: str = ".png"
    batch_size: int = 100
    num_workers: int = 8


def main():
    cfg: ComputeFIDConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ComputeFIDConfig), OmegaConf.from_cli())
    )

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    transform = transforms.Compose(
        [
            DMCrop(cfg.resolution),
            transforms.ToTensor(),
        ]
    )

    if cfg.ref_data_dir is not None:
        ref_dataset = ImageDataset(
            data_dirs=cfg.ref_data_dir,
            transform=transform,
            suffix=cfg.suffix,
            pil=True,
            return_dict=False,
        )
        ref_sampler = DistributedRangedSampler(
            ref_dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False
        )
        ref_data_loader = DataLoader(
            ref_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=True,
            sampler=ref_sampler,
        )
        ref_fid_stats = FIDStats(cfg.fid)
        for batch, _ in tqdm(ref_data_loader):
            batch = batch.cuda()
            ref_fid_stats.add_data(batch)
        mu2, sigma2 = ref_fid_stats.get_stats()
    else:
        mu2, sigma2 = None, None

    dataset = ImageDataset(
        data_dirs=cfg.data_dir,
        transform=transform,
        suffix=cfg.suffix,
        pil=True,
        return_dict=False,
    )
    sampler = DistributedRangedSampler(dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=True,
        sampler=sampler,
    )

    if is_master():
        print(f"{len(dataset)} images, {len(data_loader)} batches")
        print(data_loader.dataset.transform)

    fid_stats = FIDStats(cfg.fid)
    for batch, _ in tqdm(data_loader):
        batch = batch.cuda()
        fid_stats.add_data(batch)

    fid = fid_stats.compute_fid(mu2=mu2, sigma2=sigma2)
    if is_master():
        print(f"fid: {fid}")


if __name__ == "__main__":
    main()
