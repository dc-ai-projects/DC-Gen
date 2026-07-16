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

import os
from dataclasses import dataclass, field
from typing import Optional

import ipdb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import MISSING, OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

from dc_gen.ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, REGISTERED_SD_VAE_MODEL, AutoencoderKL
from dc_gen.aecore.models.dc_ae import DCAE
from dc_gen.apps.data_provider.sampler import DistributedRangedSampler
from dc_gen.apps.utils.config import get_config
from dc_gen.apps.utils.dist import (
    dist_barrier,
    dist_init,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_master,
    sync_tensor,
)
from dc_gen.apps.utils.image import CustomImageFolder, DMCrop
from dc_gen.c2icore.autoencoder import Autoencoder, AutoencoderConfig
from dc_gen.models.utils.network import get_dtype_from_str


@dataclass
class GenerateLatentConfig:
    image_root_path: str = MISSING
    latent_root_path: str = MISSING
    results_path: Optional[str] = None
    resolution: int = MISSING

    model_name: str = MISSING
    dtype: str = "fp32"
    scaling_factor: Optional[float] = None
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    batch_size: int = 64
    num_workers: int = 8

    task_id: int = 0
    num_samples_per_task: Optional[int] = None
    resume: bool = True

    debug: bool = False


def image_path_to_latent_path(image_path: str, image_root_path: str, latent_root_path: str) -> str:
    relative_image_path = os.path.relpath(image_path, image_root_path)
    last_dot_pos = relative_image_path.rfind(".")
    relative_npz_path = relative_image_path[:last_dot_pos] + ".npy"
    latent_path = os.path.join(latent_root_path, relative_npz_path)
    return latent_path


def main():
    torch.set_grad_enabled(False)
    cfg = get_config(GenerateLatentConfig)

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    device = torch.device("cuda")
    dtype = get_dtype_from_str(cfg.dtype)
    if cfg.model_name == "autoencoder":
        model = Autoencoder(cfg.autoencoder)
        cfg.scaling_factor = 1.0
    elif cfg.model_name in REGISTERED_DCAE_MODEL:
        if REGISTERED_DCAE_MODEL[cfg.model_name][1] is None:
            model = DCAE_HF.from_pretrained(f"mit-han-lab/{cfg.model_name}")
        else:
            dc_ae_cfg = REGISTERED_DCAE_MODEL[cfg.model_name][0](
                cfg.model_name, REGISTERED_DCAE_MODEL[cfg.model_name][1]
            )
            model = DCAE(dc_ae_cfg)
        assert cfg.scaling_factor is not None
    elif cfg.model_name in REGISTERED_SD_VAE_MODEL:
        model = REGISTERED_SD_VAE_MODEL[cfg.model_name][0](cfg.model_name, REGISTERED_SD_VAE_MODEL[cfg.model_name][1])
        assert cfg.scaling_factor is not None
    elif cfg.model_name in ["stabilityai/sd-vae-ft-ema", "flux-vae", "stabilityai/sdxl-vae", "sd3-vae"]:
        model = AutoencoderKL(cfg.model_name)
        cfg.scaling_factor = model.model.config.scaling_factor
    else:
        raise ValueError(f"{cfg.model_name} is not supported for generating latent")

    model = model.eval().to(device=device, dtype=dtype)

    if cfg.model_name in ["dinov2-vit-b", "dinov2-vit-l", "dinov2-vit-g"]:
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

        transform = transforms.Compose(
            [
                DMCrop(cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                DMCrop(cfg.resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5, inplace=True),
            ]
        )

    dataset = CustomImageFolder(cfg.image_root_path, transform, return_dict=True)

    if cfg.num_samples_per_task is not None:
        num_tasks = (len(dataset) - 1) // cfg.num_samples_per_task + 1
        print(f"num_tasks {num_tasks}")
        start, end = min(cfg.num_samples_per_task * cfg.task_id, len(dataset)), min(
            cfg.num_samples_per_task * (cfg.task_id + 1), len(dataset)
        )
        indices = list(range(start, end))
        dataset = torch.utils.data.Subset(dataset, indices)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        sampler=DistributedRangedSampler(dataset, num_replicas=get_dist_size(), rank=get_dist_rank(), shuffle=False),
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=False,
    )

    if is_master():
        os.makedirs(cfg.latent_root_path, exist_ok=cfg.resume or cfg.num_samples_per_task is not None)
    dist_barrier()

    latent_total_sum = 0
    latent_total_sum_squared = 0
    latent_total_cnt = 0

    for batch_idx, input_dict in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        disable=not is_master(),
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ):
        skip = False
        if cfg.resume:
            skip = True
            for image_path in input_dict["image_path"]:
                latent_path = image_path_to_latent_path(image_path, cfg.image_root_path, cfg.latent_root_path)
                try:
                    data = np.load(latent_path)
                    assert data.shape[1] == data.shape[2] == cfg.resolution // model.spatial_compression_ratio
                except Exception:
                    skip = False
        if skip:
            if is_master():
                print(f"skip batch {batch_idx}")
            continue

        images = input_dict["image"].cuda()
        latents = model.encode(images)
        latents = latents * cfg.scaling_factor
        latent_total_sum += latents.float().sum().item()
        latent_total_sum_squared += latents.float().square().sum().item()
        latent_total_cnt += latents.numel()
        if cfg.debug:
            outputs = model.decode(latents / cfg.scaling_factor)
            save_image(torch.cat([images[0:4], outputs[0:4]], dim=3) * 0.5 + 0.5, "tmp.jpg", nrow=2)
            ipdb.set_trace()
        for i, (image_path, _) in enumerate(zip(input_dict["image_path"], input_dict["label"])):
            latent = latents[i].cpu().numpy()
            latent_path = image_path_to_latent_path(image_path, cfg.image_root_path, cfg.latent_root_path)
            os.makedirs(os.path.dirname(latent_path), exist_ok=True)
            np.save(latent_path, latent)

    latent_total_sum = sync_tensor(torch.tensor(latent_total_sum).cuda(), reduce="sum").cpu().numpy()
    latent_total_sum_squared = sync_tensor(torch.tensor(latent_total_sum_squared).cuda(), reduce="sum").cpu().numpy()
    latent_total_cnt = sync_tensor(torch.tensor(latent_total_cnt).cuda(), reduce="sum").cpu().numpy()
    mean = latent_total_sum / latent_total_cnt
    rms = np.sqrt(latent_total_sum_squared / latent_total_cnt)
    variance = (latent_total_sum_squared - mean * latent_total_sum) / (latent_total_cnt - 1)
    std = np.sqrt(variance)
    if is_master():
        print(f"mean: {mean}, rms: {rms}, std: {std}")

    if cfg.results_path is not None and is_master():
        os.makedirs(cfg.results_path, exist_ok=True)
        with open(os.path.join(cfg.results_path, f"{cfg.num_samples_per_task}_{cfg.task_id}.txt"), "w") as f:
            f.write("complete!")


if __name__ == "__main__":
    main()
