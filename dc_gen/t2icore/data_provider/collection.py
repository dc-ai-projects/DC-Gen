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
from functools import partial

import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from ...apps.utils.dist import dist_init
from .base_train import T2ICoreDataProvider, T2ICoreLatentDataset, T2ICoreTrainDataProviderConfig

possible_datasets: dict[str, type[Dataset]] = {
}


possible_train_data_providers: dict[str, tuple[type[T2ICoreTrainDataProviderConfig], type[T2ICoreDataProvider]]] = {
}
