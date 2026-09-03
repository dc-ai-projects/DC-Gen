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

from ...apps.utils.video import MultiResolutionVideoDataset, MultiResolutionVideoFolder
from .activitynet import (
    ActivityNet12TestDataProviderConfig,
    ActivityNet13TestDataProviderConfig,
    ActivityNet13TestLongVideoDataProviderConfig,
    ActivityNetDataProvider,
)
from .dummy import DummyDataProvider, DummyDataProviderConfig, DummyDataset
from .kinetics_600 import Kinetics600DataProvider, Kinetics600Dataset, Kinetics600Test5000DataProviderConfig

possible_datasets = {
    "ActivityNet12Test": partial(MultiResolutionVideoDataset, root="~/dataset/activitynet/v1-2/test"),
    "ActivityNet13Test": partial(MultiResolutionVideoDataset, root="~/dataset/activitynet/v1-3/test"),
    "Dummy": DummyDataset,
    "Kinetics600Test": partial(
        Kinetics600Dataset, data_dir="~/dataset/kinetics_600/test", meta_path="assets/data/meta/kinetics_600_test.json"
    ),
    "UCF101": partial(MultiResolutionVideoFolder, root="~/dataset/ucf101/mp4"),
}


possible_video_eval_data_providers = {
    "ActivityNet12Test": (ActivityNet12TestDataProviderConfig, ActivityNetDataProvider),
    "ActivityNet13Test": (ActivityNet13TestDataProviderConfig, ActivityNetDataProvider),
    "ActivityNet13TestLongVideo": (ActivityNet13TestLongVideoDataProviderConfig, ActivityNetDataProvider),
    "Kinetics600Test5000": (Kinetics600Test5000DataProviderConfig, Kinetics600DataProvider),
}


possible_video_train_data_providers = {
    "Dummy": (DummyDataProviderConfig, DummyDataProvider),
}
