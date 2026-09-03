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
from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset, Sampler

from ...apps.data_provider.dc_base import BaseDataProvider, BaseDataProviderConfig
from ...apps.data_provider.sampler import DistributedRangedAspectRatioBatchSampler, DistributedRangedSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512,
    AspectRatioManager512F32MS,
    AspectRatioManager512F64MS,
    AspectRatioManager1024,
    AspectRatioManager2048,
    AspectRatioManagerVideo480F32MS,
)
from ...imageeditcore.data_provider.base_train import ImageEditCoreLatentDataset
from ...t2icore.data_provider.base_train import T2ICoreLatentDataset
from ...videogencore.data_provider.base_train import VideoGenCoreLatentDataset

__all__ = [
    "DCAdaptDataProviderConfig",
    "DCAdaptDataProvider",
    "DCAdaptLatentDataProviderConfig",
    "DCAdaptLatentDataProvider",
]


@dataclass
class DCAdaptDataProviderConfig(BaseDataProviderConfig):
    pass


class DCAdaptDataProvider(BaseDataProvider):
    def __init__(self, cfg: DCAdaptDataProviderConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptDataProviderConfig

    def build_dataset(self) -> Dataset:
        raise NotImplementedError

    def build_sampler(self) -> Sampler:
        return DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
        )


@dataclass
class DCAdaptLatentDataProviderConfig(BaseDataProviderConfig):
    save_checkpoint_steps: int = 100
    resolution: str = "512"

    shuffle_chunk_size: Optional[int] = 1000
    drop_last: bool = True
    temperature: float = 0.1

    teacher_wds_meta_dir: str = MISSING
    teacher_wds_meta_filename: str = MISSING
    student_wds_meta_dir: str = MISSING
    student_wds_meta_filename: str = MISSING

    data_ext: str = ".npy"
    image_ext: str = ".jpg"


class DCAdaptAlignLatentDataset(Dataset):
    def __init__(self, cfg: DCAdaptLatentDataProviderConfig) -> None:
        super().__init__()

        self.cfg = cfg

        self.dataset1 = T2ICoreLatentDataset(
            meta_path=os.path.join(self.cfg.teacher_wds_meta_dir, self.cfg.teacher_wds_meta_filename),
            temperature=self.cfg.temperature,
            data_ext=self.cfg.data_ext,
        )
        self.dataset2 = T2ICoreLatentDataset(
            meta_path=os.path.join(self.cfg.student_wds_meta_dir, self.cfg.student_wds_meta_filename),
            temperature=self.cfg.temperature,
            data_ext=self.cfg.data_ext,
        )

        assert len(self.dataset1) == len(self.dataset2), f"Dataset size mismatch"  # check datasets

    def __len__(self) -> int:
        return len(self.dataset1)

    def __getitem__(self, index: tuple[int, int]) -> dict[str, torch.Tensor]:
        sample1, sample2 = self.dataset1[index], self.dataset2[index]
        data1, caption1 = sample1["images"], sample1["captions"]
        data2, caption2 = sample2["images"], sample2["captions"]
        height1, width1 = sample1["height"], sample1["width"]
        height2, width2 = sample2["height"], sample2["width"]

        # Caption must be the same under the identical sampling seed.
        assert caption1 == caption2, f"Caption mismatch: {caption1} != {caption2}"
        assert (
            height1 * width2 == height2 * width1
        ), f"Proportion mismatch: {height1} : {width1} v.s. {height2} : {width2}"

        return {
            "data1": data1,
            "data2": data2,
            "caption": caption1,
            "height": height1,
            "width": width1,
        }

    def get_data_info(self, index: tuple[int, int]):
        sample = self.__getitem__(index)
        return {
            "height": sample["height"],
            "width": sample["width"],
        }


class DCAdaptAlignImageEditLatentDataset(DCAdaptAlignLatentDataset):
    def __init__(self, cfg: DCAdaptLatentDataProviderConfig) -> None:
        self.cfg = cfg

        self.dataset1 = ImageEditCoreLatentDataset(
            meta_path=os.path.join(self.cfg.teacher_wds_meta_dir, self.cfg.teacher_wds_meta_filename),
            temperature=self.cfg.temperature,
            data_ext=self.cfg.data_ext,
            image_ext=self.cfg.image_ext,
        )
        self.dataset2 = ImageEditCoreLatentDataset(
            meta_path=os.path.join(self.cfg.student_wds_meta_dir, self.cfg.student_wds_meta_filename),
            temperature=self.cfg.temperature,
            data_ext=self.cfg.data_ext,
            image_ext=self.cfg.image_ext,
        )

        assert len(self.dataset1) == len(self.dataset2), f"Dataset size mismatch"  # check datasets

    def __getitem__(self, index: tuple[int, int]) -> dict[str, torch.Tensor]:
        sample1, sample2 = self.dataset1[index], self.dataset2[index]
        data1, caption1 = sample1["images"], sample1["captions"]
        data2, caption2 = sample2["images"], sample2["captions"]

        if "ae_features" in sample1:
            assert "ae_features" in sample2
            ae_feature1, ae_feature2 = sample1["ae_features"], sample2["ae_features"]

        height1, width1 = int(sample1["height"]), int(sample1["width"])
        height2, width2 = int(sample2["height"]), int(sample2["width"])

        # Caption must be the same under the identical sampling seed.
        assert caption1 == caption2, f"Caption mismatch: {caption1} != {caption2}"
        assert (
            height1 * width2 == height2 * width1
        ), f"Proportion mismatch: {height1} : {width1} v.s. {height2} : {width2}"

        return_dict = {
            "data1": data1,
            "data2": data2,
            "caption": caption1,
            "height": height1,
            "width": width1,
        }

        if "ae_features" in sample1:
            return_dict["ae_feature1"] = ae_feature1
            return_dict["ae_feature2"] = ae_feature2

        return return_dict


class DCAdaptAlignVideoLatentDataset(DCAdaptAlignLatentDataset):
    def __init__(self, cfg: DCAdaptLatentDataProviderConfig) -> None:
        self.cfg = cfg

        self.dataset1 = VideoGenCoreLatentDataset(
            meta_path=os.path.join(self.cfg.teacher_wds_meta_dir, self.cfg.teacher_wds_meta_filename),
            temperature=self.cfg.temperature,
            latent_ext=self.cfg.data_ext,
        )
        self.dataset2 = VideoGenCoreLatentDataset(
            meta_path=os.path.join(self.cfg.student_wds_meta_dir, self.cfg.student_wds_meta_filename),
            temperature=self.cfg.temperature,
            latent_ext=self.cfg.data_ext,
        )

        assert len(self.dataset1) == len(self.dataset2), f"Dataset size mismatch"  # check datasets

    def __getitem__(self, index: tuple[int, int]) -> dict[str, torch.Tensor]:
        sample1, sample2 = self.dataset1[index], self.dataset2[index]
        data1, caption1 = sample1["videos"], sample1["captions"]
        data2, caption2 = sample2["videos"], sample2["captions"]

        if "ae_feature" in sample1:
            assert "ae_feature" in sample2
            ae_feature1, ae_feature2 = sample1["ae_feature"], sample2["ae_feature"]

        height1, width1 = int(sample1["height"]), int(sample1["width"])
        height2, width2 = int(sample2["height"]), int(sample2["width"])

        # Caption must be the same under the identical sampling seed.
        assert caption1 == caption2, f"Caption mismatch: {caption1} != {caption2}"
        assert (
            height1 * width2 == height2 * width1
        ), f"Proportion mismatch: {height1} : {width1} v.s. {height2} : {width2}"

        return_dict = {
            "data1": data1,
            "data2": data2,
            "caption": caption1,
            "height": height1,
            "width": width1,
        }

        if "ae_feature" in sample1:
            return_dict["ae_feature1"] = ae_feature1
            return_dict["ae_feature2"] = ae_feature2

        return return_dict


class DCAdaptLatentDataProvider(BaseDataProvider):
    def __init__(self, cfg: DCAdaptLatentDataProviderConfig):
        if cfg.resolution == "480F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F32MS()
        elif cfg.resolution == "512":
            self.aspect_ratio_manager = AspectRatioManager512()
        elif cfg.resolution == "512F32MS":
            self.aspect_ratio_manager = AspectRatioManager512F32MS()
        elif cfg.resolution == "512F64MS":
            self.aspect_ratio_manager = AspectRatioManager512F64MS()
        elif cfg.resolution == "1024":
            self.aspect_ratio_manager = AspectRatioManager1024()
        elif cfg.resolution == "2048":
            self.aspect_ratio_manager = AspectRatioManager2048()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported for SanaMSCrop")

        super().__init__(cfg)
        self.cfg: DCAdaptLatentDataProviderConfig

    def __len__(self):
        return len(self.dataset)

    def build_dataset(self) -> Dataset:
        dataset = DCAdaptAlignLatentDataset(cfg=self.cfg)
        return dataset

    def build_sampler(self) -> DistributedRangedAspectRatioBatchSampler:
        raw_sampler = DistributedRangedSampler(
            self.dataset,
            self.dist_size,
            self.rank,
            shuffle=self.cfg.shuffle,
            seed=self.cfg.seed,
            drop_last=self.cfg.drop_last,
            shuffle_chunk_size=self.cfg.shuffle_chunk_size,
        )
        sampler = DistributedRangedAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=self.cfg.save_checkpoint_steps,
            aspect_ratio_manager=self.aspect_ratio_manager,
            drop_last=False,
        )
        return sampler

    def build_data_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            collate_fn=self.collate_fn,
            generator=generator,
        )
        return data_loader
