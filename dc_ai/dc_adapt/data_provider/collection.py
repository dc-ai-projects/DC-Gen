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

from torch.utils.data import Dataset
from tqdm import tqdm

from ...apps.utils.config import get_config
from ...apps.utils.dist import dist_init
from .base import (
    DCAdaptAlignImageEditLatentDataset,
    DCAdaptAlignLatentDataset,
    DCAdaptAlignVideoLatentDataset,
    DCAdaptLatentDataProvider,
    DCAdaptLatentDataProviderConfig,
)
from .latent_fusionx import DCAdaptLatentFusionXDataProvider, DCAdaptLatentFusionXDataProviderConfig
from .latent_krea_gen import DCAdaptLatentFluxReasonDataProviderConfig
from .latent_pico_banana import (
    DCAdaptLatentPicoBananaDataProvider,
    DCAdaptLatentPicoBananaDataProviderConfig,
    DCAdaptLatentPicoBananaQwenImageGenDataProviderConfig,
)

possible_datasets: dict[str, tuple[type[DCAdaptLatentDataProviderConfig], type[Dataset]]] = {
    "LatentFusionXAlign": (DCAdaptLatentFusionXDataProviderConfig, DCAdaptAlignVideoLatentDataset),
    "LatentPicoBananaAlign": (DCAdaptLatentPicoBananaDataProviderConfig, DCAdaptAlignImageEditLatentDataset),
    "LatentPicoBananaQwenImageGenAlign": (
        DCAdaptLatentPicoBananaQwenImageGenDataProviderConfig,
        DCAdaptAlignImageEditLatentDataset,
    ),
}


possible_train_data_providers: dict[
    str, tuple[type[DCAdaptLatentDataProviderConfig], type[DCAdaptLatentDataProvider]]
] = {
    "LatentFluxReasonAlign": (DCAdaptLatentFluxReasonDataProviderConfig, DCAdaptLatentDataProvider),
    "LatentFusionXAlign": (DCAdaptLatentFusionXDataProviderConfig, DCAdaptLatentFusionXDataProvider),
    "LatentPicoBananaAlign": (DCAdaptLatentPicoBananaDataProviderConfig, DCAdaptLatentPicoBananaDataProvider),
    "LatentPicoBananaQwenImageGenAlign": (
        DCAdaptLatentPicoBananaQwenImageGenDataProviderConfig,
        DCAdaptLatentPicoBananaDataProvider,
    ),
}
