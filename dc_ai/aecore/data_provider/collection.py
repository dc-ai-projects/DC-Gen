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
from functools import partial
from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm

from ...apps.utils.config import get_config
from ...apps.utils.dist import dist_init
from ...apps.utils.image import IdentityTransform, MultiResolutionImageDataset, MultiResolutionImageFolder
from .base import AECoreDataProvider, AECoreDataProviderConfig
from .imagenet import (
    ImageNetDataProvider,
    ImageNetEvalDataProviderConfig,
    ImageNetTrainDataProviderConfig,
)
from .mjhq import MJHQDataProvider, MJHQEvalDataProviderConfig

possible_datasets: dict[str, Callable[..., Dataset]] = {
    "ImageNet_eval": partial(MultiResolutionImageFolder, root="~/dataset/imagenet/val"),
    "ImageNet_train": partial(MultiResolutionImageFolder, root="~/dataset/imagenet/train"),
}


possible_eval_data_providers: dict[str, tuple[type[AECoreDataProviderConfig], type[AECoreDataProvider]]] = {
    "ImageNet": (ImageNetEvalDataProviderConfig, ImageNetDataProvider),
    "MJHQ": (MJHQEvalDataProviderConfig, MJHQDataProvider),
}


possible_train_data_providers: dict[str, tuple[type[AECoreDataProviderConfig], type[AECoreDataProvider]]] = {
    "ImageNet": (ImageNetTrainDataProviderConfig, ImageNetDataProvider),
}


@dataclass
class DataProviderCollectionConfig:
    load_all_data: bool = False
    save_train_examples_path: Optional[str] = None


def main():
    cfg = get_config(DataProviderCollectionConfig)
    dist_init()
    for dataset_name, dataset_class in possible_datasets.items():
        try:
            dataset = dataset_class(size_transform=IdentityTransform(), transform=IdentityTransform())
            print(f"{dataset_name}: {len(dataset)}")
            if cfg.load_all_data:
                for i in tqdm(range(0, len(dataset), 1000)):
                    dataset[i]
        except Exception as e:
            print(f"failed to load {dataset_name}: {e}")

    if cfg.save_train_examples_path is not None:
        os.makedirs(cfg.save_train_examples_path, exist_ok=True)
        for data_provider_name, (
            data_provider_config_class,
            data_provider_class,
        ) in possible_train_data_providers.items():
            try:
                data_provider_cfg = data_provider_config_class(min_h=256, min_w=256, batch_size=64)
                data_provider = data_provider_class(data_provider_cfg)
                batch = next(iter(data_provider.data_loader))
                save_image(
                    batch["image"] * 0.5 + 0.5, os.path.join(cfg.save_train_examples_path, f"{data_provider_name}.jpg")
                )
            except Exception as e:
                print(f"warning: failed to save examples for {data_provider_name}: {e}")

    train_data_provider_statistics = {}
    for _, (data_provider_config_class, data_provider_class) in possible_train_data_providers.items():
        for resolution in [256, 512, 1024]:
            data_provider_cfg = data_provider_config_class(min_h=resolution, min_w=resolution)
            try:
                data_provider = data_provider_class(data_provider_cfg)
                if f"{data_provider_cfg.name}" not in train_data_provider_statistics:
                    train_data_provider_statistics[f"{data_provider_cfg.name}"] = {}
                train_data_provider_statistics[f"{data_provider_cfg.name}"][f"{resolution}"] = len(
                    data_provider.dataset
                )
            except Exception as e:
                data_provider = None
                print(f"warning: failed to compute statistics for dataset {data_provider_cfg.name}: {e}")
        if data_provider is not None:
            metadata = data_provider.metadata
            train_data_provider_statistics[f"{data_provider_cfg.name}"]["possible_modes"] = np.unique(
                metadata["mode"]
            ).tolist()
    for key, value in train_data_provider_statistics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

"""
python -m dc_gen.aecore.data_provider.collection
ImageNet_eval: 50000
ImageNet_train: 1281167

ImageNetTrain: {'256': 1152197, '512': 68410, '1024': 12635, 'possible_modes': ['RGB']}
"""
