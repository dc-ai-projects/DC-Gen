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

import io
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
from omegaconf import MISSING
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...aecore.autoencoder import Autoencoder, AutoencoderConfig
from ...apps.data_provider.generate_latent_archives import (
    LatentArchivesGenerator,
    LatentArchivesGeneratorConfig,
    SingleTarDataset,
    convert_image_to_rgb,
)
from ...apps.data_provider.sampler import AspectRatioBatchSampler, DistributedRangedSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512F32MS,
    AspectRatioManager512F64MS,
    AspectRatioManager1024,
    AspectRatioManager2048,
    AspectRatioManager4096,
)
from ...apps.utils.config import get_config
from ...apps.utils.image import QwenImageResize


@dataclass
class ImageEditCoreLatentArchivesGeneratorConfig(LatentArchivesGeneratorConfig):
    dataset_name: str = MISSING

    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    resolution: str = "512F32"
    size_transform: str = "QwenImageResize"
    batch_sampler: str = "AspectRatioBatchSampler"


class ImageEditCoreSingleImageTarDataset(SingleTarDataset):
    def __init__(self, tar_path: str, image_ext: str, transform: Optional[Callable], dataset_name: str):
        super().__init__(tar_path)
        self.image_ext = image_ext
        self.transform = transform
        self.dataset_name = dataset_name

        if self.dataset_name in ["pico_banana"]:
            pass
        else:
            raise ValueError(f"dataset {dataset_name} is not supported")

    def get_data_info(self, index: int) -> dict:
        name, size, offset = self.sample_meta_list[index][".json"]
        stream = io.BytesIO(self._file_data[offset : offset + size])
        data_info = json.load(stream)
        if "height" not in data_info:
            sample = super().__getitem__(index)
            image = sample[self.image_ext]
            height, width = image.height, image.width // 2
            data_info["height"] = height
            data_info["width"] = width
        return data_info

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        image = sample[self.image_ext]
        image = convert_image_to_rgb(image)
        mid_x = image.width // 2
        input_image = image.crop((0, 0, mid_x, image.height))
        edited_image = image.crop((mid_x, 0, image.width, image.height))

        if "height" not in sample[".json"]:
            image = sample[self.image_ext]
            height, width = image.height, image.width // 2
            sample[".json"]["height"] = height
            sample[".json"]["width"] = width

        if self.transform is not None:
            sample[self.image_ext] = input_image
            input_image = self.transform(input_image)
            edited_image = self.transform(edited_image)

        sample["input_image"] = input_image
        sample["image"] = edited_image

        captions = []
        clip_scores = []
        corrupted = False

        if self.dataset_name in ["pico_banana"]:
            if "captions" in sample[".json"] and "clip_scores" in sample[".json"]:
                captions = sample[".json"]["captions"]
                clip_scores = sample[".json"]["clip_scores"]
            else:
                captions, clip_scores = [""], ["30.0"]
                corrupted = True

        else:
            raise ValueError(f"dataset {self.dataset_name} is not supported")

        sample[".json"]["corrupted"] = corrupted
        sample[".json"]["captions"] = captions
        sample[".json"]["clip_scores"] = clip_scores
        return sample


class ImageEditCoreLatentArchivesGenerator(LatentArchivesGenerator):
    def __init__(self, cfg: ImageEditCoreLatentArchivesGeneratorConfig):
        if cfg.resolution == "512F32MS":
            self.aspect_ratio_manager = AspectRatioManager512F32MS()
        elif cfg.resolution == "512F64MS":
            self.aspect_ratio_manager = AspectRatioManager512F64MS()
        elif cfg.resolution == "1024":
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == "2048":
            self.aspect_ratio_manager = AspectRatioManager2048()
        elif cfg.resolution == "4096":
            self.aspect_ratio_manager = AspectRatioManager4096()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for SanaMSCrop")

        super().__init__(cfg)
        self.cfg: ImageEditCoreLatentArchivesGeneratorConfig

        self.transform = self.build_size_transform()

    def build_model(self) -> nn.Module:
        model = Autoencoder(self.cfg.autoencoder).to(device=self.device, dtype=self.dtype)
        return model

    def build_size_transform(self) -> Callable:
        if self.cfg.size_transform == "QwenImageResize":
            if self.cfg.resolution in ["512F32MS", "512F64MS"]:
                resolution = 512
            elif self.cfg.resolution in ["1024", "2048", "4096"]:
                resolution = int(self.cfg.resolution)
            else:
                raise ValueError(f"Resolution {self.cfg.resolution} is not supported")
            if self.cfg.autoencoder.name in ["qwen-vae", "flux-vae", "dc-ae-f32c32-in-1.0"]:
                total_compression_ratio = 32
            elif self.cfg.autoencoder.name in ["dc-ae-f64c128-in-1.0"]:
                total_compression_ratio = 64
            size_transform = QwenImageResize(
                resolution=resolution,
                total_compression_ratio=total_compression_ratio,
            )
        else:
            raise ValueError(
                f"Image Edit Core must use QwenImageResize as size transformation. Current transform: {self.cfg.size_transform}"
            )

        return size_transform

    def build_dataset_for_single_archive(self, index: int) -> ImageEditCoreSingleImageTarDataset:
        dataset = ImageEditCoreSingleImageTarDataset(
            self.shardlist[index]["url"],
            self.cfg.image_ext,
            self.transform,
            self.cfg.dataset_name,
        )
        return dataset

    def build_data_loader_for_single_archive(self, index: int) -> DataLoader:
        dataset = self.build_dataset_for_single_archive(index)
        self.sampler = DistributedRangedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        if self.cfg.batch_sampler == "AspectRatioBatchSampler":
            self.batch_sampler = AspectRatioBatchSampler(
                sampler=self.sampler,
                dataset=dataset,
                batch_size=self.cfg.batch_size,
                save_checkpoint_steps=None,
                aspect_ratio_manager=self.aspect_ratio_manager,
                drop_last=False,
            )
        else:
            raise ValueError(f"batch sampler {self.cfg.batch_sampler} is not supported")
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
        )
        return data_loader

    def generate_latent(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            latent = self.model.encode(image)
        return latent

    def generate_data_for_single_archive(self, index: int):
        data_loader = self.build_data_loader_for_single_archive(index)

        data_format = {self.cfg.latent_ext: "data", ".json": "data", ".jpg": "data"}
        data_list = []

        for batch in tqdm(data_loader, desc=f"Generating latent for {self.shardlist[index]['url']}"):
            image, input_image = batch["image"], batch["input_image"]
            image = image.to(device=self.device, dtype=self.dtype)
            input_image = input_image.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                latent = self.generate_latent(image)
                ae_feature = self.generate_latent(input_image)
            for i in range(image.shape[0]):
                if self.cfg.latent_ext == ".pth":
                    sample = {
                        ".pth": {
                            "latent": latent[i].cpu().to(dtype=self.latent_dtype),
                            "ae_feature": ae_feature[i].cpu().to(dtype=self.latent_dtype),
                        }
                    }
                else:
                    raise ValueError(f"latent ext {self.cfg.latent_ext} is not supported")
                for key, value in batch.items():
                    if key.startswith("."):
                        sample[key] = value[i]
                data_list.append((batch["__key__"][i], sample))

        new_data_list = []
        for data in data_list:
            if not data[1][".json"]["corrupted"]:
                data[1][".json"].pop("corrupted")
                new_data_list.append(data)

        print(f"Found {len(new_data_list)}/{len(data_list)} valid samples in archive {index}")

        return data_format, new_data_list


def main():
    cfg: ImageEditCoreLatentArchivesGeneratorConfig = get_config(ImageEditCoreLatentArchivesGeneratorConfig)
    generator = ImageEditCoreLatentArchivesGenerator(cfg)
    generator.generate()


if __name__ == "__main__":
    main()
