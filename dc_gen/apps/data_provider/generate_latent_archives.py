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

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import ipdb
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from omegaconf import MISSING
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, default_collate
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ...models.utils.network import get_dtype_from_str
from ..utils.image import DMCrop, convert_image_to_rgb
from ..utils.tar import SingleTarDataset
from ..utils.zip import SingleZipDataset, single_zip_dataset_worker_init_fn
from .generate_archives import ArchivesGenerator, ArchivesGeneratorConfig


@dataclass
class LatentArchivesGeneratorConfig(ArchivesGeneratorConfig):
    meta_path: str = MISSING
    original_archive_dir: str = MISSING
    resolution: Optional[int] = None
    size_transform: str = "DMCrop"
    mean: float = 0.5
    std: float = 0.5

    dtype: str = "fp32"

    batch_size: int = 64
    num_workers: int = 8

    image_ext: str = ".jpg"
    latent_dtype: str = "fp32"
    latent_ext: str = ".npy"

    load_archive_format: str = "${.archive_format}"  # "tar" or "zip" or "parquet"


class SingleImageTarDataset(SingleTarDataset):
    def __init__(self, tar_path: str, image_ext: str, transform: Optional[Callable]):
        super().__init__(tar_path)
        self.image_ext = image_ext
        self.transform = transform

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        image = sample[self.image_ext]
        if self.transform is not None:
            image = convert_image_to_rgb(image)
            image = self.transform(image)
        del sample[self.image_ext]
        sample["image"] = image
        return sample


class SingleImageZipDataset(SingleZipDataset):
    def __init__(self, zip_path: str, image_ext: str, transform: Optional[Callable]):
        super().__init__(zip_path)
        self.image_ext = image_ext
        self.transform = transform

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        image = sample[self.image_ext]
        if self.transform is not None:
            image = convert_image_to_rgb(image)
            image = self.transform(image)
        del sample[self.image_ext]
        sample["image"] = image
        return sample


class LatentArchivesGenerator(ArchivesGenerator):
    def __init__(self, cfg: LatentArchivesGeneratorConfig):
        super().__init__(cfg)
        self.cfg: LatentArchivesGeneratorConfig
        self.dtype = get_dtype_from_str(cfg.dtype)
        self.model = self.build_model()
        if cfg.load_archive_format in ["tar", "zip", "parquet"]:
            with open(self.cfg.meta_path, "r") as f:
                self.meta = json.load(f)
            self.shardlist = self.meta["shardlist"]
        else:
            raise ValueError(f"Unsupported load archive format: {cfg.load_archive_format}")
        self.transform = transforms.Compose(
            [
                self.build_size_transform(),
                transforms.ToTensor(),
                transforms.Normalize(cfg.mean, cfg.std),
            ]
        )
        self.latent_dtype = get_dtype_from_str(cfg.latent_dtype)

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def build_size_transform(self) -> Callable:
        if self.cfg.size_transform == "DMCrop":
            size_transform = DMCrop(self.cfg.resolution)
        else:
            raise ValueError(f"size transform {self.cfg.size_transform} is not supported")
        return size_transform

    def get_save_path(self, index: int) -> str:
        if self.cfg.archive_format == "tar":
            original_tar_path = self.shardlist[index]["url"]
            relative_tar_path = os.path.relpath(original_tar_path, self.cfg.original_archive_dir)
            save_path = os.path.join(self.cfg.save_dir, relative_tar_path)
            # change the extension to .tar
            save_path = os.path.splitext(save_path)[0] + ".tar"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            return save_path
        elif self.cfg.archive_format == "zip":
            original_tar_path = self.shardlist[index]["url"]
            relative_tar_path = os.path.relpath(original_tar_path, self.cfg.original_archive_dir)
            save_path = os.path.join(self.cfg.save_dir, relative_tar_path)
            # change the extension to .zip
            save_path = os.path.splitext(save_path)[0] + ".zip"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            return save_path
        else:
            raise ValueError(f"Unsupported archive format: {self.cfg.archive_format}")

    def build_dataset_for_single_archive(self, index: int) -> Dataset:
        if self.cfg.load_archive_format == "tar":
            dataset = SingleImageTarDataset(self.shardlist[index]["url"], self.cfg.image_ext, self.transform)
        elif self.cfg.load_archive_format == "zip":
            dataset = SingleImageZipDataset(self.shardlist[index]["url"], self.cfg.image_ext, self.transform)
        else:
            raise NotImplementedError(f"Unsupported load archive format: {self.cfg.load_archive_format}")
        return dataset

    def collate_fn(self, batch):
        image = default_collate([item["image"] for item in batch])
        batch = {key: [item[key] for item in batch] for key in batch[0] if key != "image"}
        batch["image"] = image
        return batch

    def build_data_loader_for_single_archive(self, index: int) -> DataLoader:
        dataset = self.build_dataset_for_single_archive(index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            worker_init_fn=single_zip_dataset_worker_init_fn if self.cfg.load_archive_format == "zip" else None,
        )
        return data_loader

    def generate_latent(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def generate_data_for_single_archive(self, index: int):
        data_loader = self.build_data_loader_for_single_archive(index)

        data_format = {self.cfg.latent_ext: "data", ".json": "data"}
        data_list = []

        for batch in tqdm(data_loader, desc=f"Generating latent for {self.shardlist[index]['url']}"):
            image = batch["image"]
            image = image.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                latent = self.generate_latent(image)
            for i in range(image.shape[0]):
                if self.cfg.latent_ext == ".npy":
                    sample = {".npy": latent[i].cpu().numpy()}
                elif self.cfg.latent_ext == ".pth":
                    sample = {".pth": latent[i].cpu().to(dtype=self.latent_dtype)}
                else:
                    raise ValueError(f"latent ext {self.cfg.latent_ext} is not supported")
                for key, value in batch.items():
                    if key.startswith("."):
                        sample[key] = value[i]
                data_list.append((batch["__key__"][i], sample))

        return data_format, data_list
