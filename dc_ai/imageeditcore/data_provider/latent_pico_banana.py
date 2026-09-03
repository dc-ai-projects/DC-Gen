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

from dataclasses import dataclass

from .base_train import ImageEditCoreLatentTrainDataProvider, ImageEditCoreLatentTrainDataProviderConfig


@dataclass
class LatentPicoBananaTrainDataProviderConfig(ImageEditCoreLatentTrainDataProviderConfig):
    name: str = "LatentPicoBanana"


@dataclass
class LatentPicoBananaSingleTrainDataProviderConfig(LatentPicoBananaTrainDataProviderConfig):
    name: str = "LatentPicoBananaSingle"
    wds_meta_filename: str = "pico_banana_single.json"


@dataclass
class LatentPicoBananaQwenImageGenTrainDataProviderConfig(LatentPicoBananaTrainDataProviderConfig):
    name: str = "LatentPicoBananaQwenImageGen"
    wds_meta_filename: str = "pico_banana_qwen_image_gen.json"


class LatentPicoBananaDataProvider(ImageEditCoreLatentTrainDataProvider):
    def __init__(self, cfg: LatentPicoBananaTrainDataProviderConfig):
        super().__init__(cfg)
        self.cfg: LatentPicoBananaTrainDataProviderConfig
