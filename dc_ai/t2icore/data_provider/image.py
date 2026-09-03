# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Image data providers for on-the-fly latent encoding."""

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from ...apps.data_provider.sampler import DistributedRangedAspectRatioBatchSampler, DistributedRangedSampler
from ...apps.data_provider.web_dataset.wids import WebDataset
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512,
    AspectRatioManager1024,
    AspectRatioManager2048,
    BaseAspectRatioManager,
)
from ...apps.utils.image import AspectRatioResizeCenterCrop, convert_image_to_rgb
from ...apps.utils.io import read_json
from .base_train import T2ICoreDataProvider, T2ICoreTrainDataProviderConfig

__all__ = [
    "T2ICoreImageDataset",
    "T2ICoreImageTrainDataProviderConfig",
    "T2ICoreImageTrainDataProvider",
    "build_image_transform",
    "get_image_captions",
    "get_image_dimensions",
    "get_image_sidecar_entry",
]

_SIDECAR_CACHE_SIZE = 10


@dataclass
class T2ICoreImageTrainDataProviderConfig(T2ICoreTrainDataProviderConfig):
    image_ext: str = ".jpg"
    mean: float = 0.5
    std: float = 0.5


def get_image_sidecar_entry(sidecar: dict[str, Any], sidecar_path: str, key: str) -> dict[str, Any]:
    if key not in sidecar:
        raise KeyError(f"Image metadata sidecar {sidecar_path} is missing sample key {key!r}")
    entry = sidecar[key]
    if not isinstance(entry, dict):
        raise ValueError(f"Sidecar entry {key!r} in {sidecar_path} must be an object")
    corrupted = entry.get("corrupted", False)
    if not isinstance(corrupted, bool):
        raise ValueError(f"Sidecar entry {key!r} in {sidecar_path} has a non-boolean corrupted field")
    return entry


def get_image_dimensions(
    entry: dict[str, Any],
    sidecar_path: str,
    key: str,
) -> Optional[dict[str, int]]:
    if entry.get("corrupted", False):
        return None
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Sidecar entry {key!r} in {sidecar_path} is missing metadata")
    height = metadata.get("H")
    width = metadata.get("W")
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
        or not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
    ):
        raise ValueError(f"Sidecar entry {key!r} in {sidecar_path} must contain positive integer metadata.H/W")
    return {"height": height, "width": width}


def get_image_captions(
    entry: dict[str, Any],
    sidecar_path: str,
    key: str,
) -> tuple[list[str], list[Any]]:
    captions = []
    clip_scores = []
    for model_name, prompt_data in entry.items():
        if model_name in ("metadata", "corrupted"):
            continue
        if not isinstance(prompt_data, dict):
            raise ValueError(f"Sidecar model {model_name!r} for sample {key!r} in {sidecar_path} must be an object")
        for prompt_kind, caption_data in prompt_data.items():
            if not isinstance(caption_data, dict):
                raise ValueError(
                    f"Sidecar caption {model_name!r}/{prompt_kind!r} for sample {key!r} "
                    f"in {sidecar_path} must be an object"
                )
            caption = caption_data.get("caption")
            if not isinstance(caption, str) or not caption:
                raise ValueError(
                    f"Sidecar caption {model_name!r}/{prompt_kind!r} for sample {key!r} "
                    f"in {sidecar_path} must contain a non-empty caption"
                )
            if "clip_score" not in caption_data:
                raise ValueError(
                    f"Sidecar caption {model_name!r}/{prompt_kind!r} for sample {key!r} "
                    f"in {sidecar_path} is missing clip_score"
                )
            clip_score = caption_data["clip_score"]
            if isinstance(clip_score, bool):
                raise ValueError(f"Sidecar caption for sample {key!r} has a boolean clip_score")
            try:
                float(clip_score)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Sidecar caption {model_name!r}/{prompt_kind!r} for sample {key!r} "
                    f"in {sidecar_path} has an invalid clip_score {clip_score!r}"
                ) from error
            captions.append(caption)
            clip_scores.append(clip_score)
    if not captions:
        raise ValueError(f"Sidecar entry {key!r} in {sidecar_path} contains no captions")
    return captions, clip_scores


def build_image_transform(
    aspect_ratio_manager: BaseAspectRatioManager,
    mean: float,
    std: float,
) -> transforms.Compose:
    return transforms.Compose(
        [
            AspectRatioResizeCenterCrop(aspect_ratio_manager),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _aspect_ratio_manager(resolution: int) -> BaseAspectRatioManager:
    if resolution == 512:
        return AspectRatioManager512()
    if resolution == 1024:
        return AspectRatioManager1024()
    if resolution == 2048:
        return AspectRatioManager2048()
    raise ValueError(f"resolution {resolution} is not supported for image training")


class T2ICoreImageDataset(WebDataset):
    """Read images and normalized sidecar metadata.

    Returned image tensors have shape [C, H, W] and values in [-1, 1] when
    the default mean and standard deviation are used.
    """

    def __init__(
        self,
        meta_path: str,
        resolution: int,
        temperature: float,
        image_ext: str,
        mean: float,
        std: float,
    ) -> None:
        super().__init__(data_dir=None, meta_path=meta_path)
        self.temperature = temperature
        self.image_ext = image_ext
        self.transform = build_image_transform(_aspect_ratio_manager(resolution), mean, std)
        self.sidecar_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _load_sidecar(self, sidecar_path: str) -> dict[str, Any]:
        if sidecar_path in self.sidecar_cache:
            sidecar = self.sidecar_cache.pop(sidecar_path)
            self.sidecar_cache[sidecar_path] = sidecar
            return sidecar
        if not os.path.isfile(sidecar_path):
            raise FileNotFoundError(f"Image metadata sidecar does not exist: {sidecar_path}")
        sidecar = read_json(sidecar_path)
        if not isinstance(sidecar, dict):
            raise ValueError(f"Image metadata sidecar must contain an object: {sidecar_path}")
        self.sidecar_cache[sidecar_path] = sidecar
        if len(self.sidecar_cache) > _SIDECAR_CACHE_SIZE:
            self.sidecar_cache.popitem(last=False)
        return sidecar

    def _load_entry(self, shard_path: str, key: str) -> tuple[dict[str, Any], str]:
        sidecar_path = shard_path.removesuffix(".tar") + ".json"
        sidecar = self._load_sidecar(sidecar_path)
        entry = get_image_sidecar_entry(sidecar, sidecar_path, key)
        return entry, sidecar_path

    def __getitem__(self, index: tuple[int, int]) -> dict[str, Any]:
        index, seed = index
        sample = self.dataset[index]
        key = sample["__key__"]
        entry, sidecar_path = self._load_entry(sample["__shard__"], key)
        dimensions = get_image_dimensions(entry, sidecar_path, key)
        if dimensions is None:
            raise RuntimeError(f"Corrupted sample {key!r} must not be loaded")
        captions, clip_scores = get_image_captions(entry, sidecar_path, key)

        weights = np.array(clip_scores, dtype=float) ** (1.0 / max(self.temperature, 0.01))
        probabilities = weights / np.sum(weights)
        random_state = np.random.RandomState(seed % 2**32)
        selected_index = random_state.choice(range(len(captions)), p=probabilities)

        image = convert_image_to_rgb(sample[self.image_ext])
        image = self.transform(image)
        return {
            "image": image,
            "captions": captions[selected_index],
            **dimensions,
        }

    def get_data_info(self, index: tuple[int, int]) -> Optional[dict[str, int]]:
        index, _ = index
        shard, inner_index, shard_description = self.dataset.get_shard(index)
        sample = shard[inner_index]
        key = sample["__key__"]
        entry, sidecar_path = self._load_entry(shard_description["url"], key)
        return get_image_dimensions(entry, sidecar_path, key)


class T2ICoreImageTrainDataProvider(T2ICoreDataProvider):
    def __init__(self, cfg: T2ICoreImageTrainDataProviderConfig):
        self.aspect_ratio_manager = _aspect_ratio_manager(cfg.resolution)
        super().__init__(cfg)
        self.cfg: T2ICoreImageTrainDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        return T2ICoreImageDataset(
            meta_path=os.path.join(self.cfg.wds_meta_dir, self.cfg.wds_meta_filename),
            resolution=self.cfg.resolution,
            temperature=self.cfg.temperature,
            image_ext=self.cfg.image_ext,
            mean=self.cfg.mean,
            std=self.cfg.std,
        )

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
        return DistributedRangedAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=self.cfg.save_checkpoint_steps,
            aspect_ratio_manager=self.aspect_ratio_manager,
            drop_last=False,
        )

    def build_data_loader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )
