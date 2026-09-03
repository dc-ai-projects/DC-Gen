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

from functools import partial

from torch.utils.data import Dataset

from .base_train import ImageEditCoreLatentDataset, ImageEditCoreTrainDataProvider, ImageEditCoreTrainDataProviderConfig
from .latent_pico_banana import (
    LatentPicoBananaDataProvider,
    LatentPicoBananaQwenImageGenTrainDataProviderConfig,
    LatentPicoBananaSingleTrainDataProviderConfig,
)

possible_datasets: dict[str, type[Dataset]] = {
    "LatentPicoBananaSingle": partial(
        ImageEditCoreLatentDataset,
        meta_path="assets/data/meta/wids/dc_ae_f32c32_in_1.0_512/pico_banana_single.json",
    ),
    "LatentPicoBananaQwenImageGen": partial(
        ImageEditCoreLatentDataset,
        meta_path="assets/data/meta/wids/dc_ae_f32c32_in_1.0_512/pico_banana_qwen_image_gen.json",
    ),
}


possible_train_data_providers: dict[
    str, tuple[type[ImageEditCoreTrainDataProviderConfig], type[ImageEditCoreTrainDataProvider]]
] = {
    "LatentPicoBananaSingle": (LatentPicoBananaSingleTrainDataProviderConfig, LatentPicoBananaDataProvider),
    "LatentPicoBananaQwenImageGen": (LatentPicoBananaQwenImageGenTrainDataProviderConfig, LatentPicoBananaDataProvider),
}
