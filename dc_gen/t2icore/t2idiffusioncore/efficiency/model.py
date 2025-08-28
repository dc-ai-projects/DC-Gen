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

from dataclasses import dataclass, field

from omegaconf import MISSING

from ...text_encoder import SingleTextEncoderConfig
from ..models.sana_t2i import SanaT2IConfig


@dataclass
class ModelConfig:
    model: str = MISSING
    sana_t2i: SanaT2IConfig = field(default_factory=SanaT2IConfig)
    batch_size: int = 2
    opset: int = 17
    large: bool = False
    text_encoders: dict[str, SingleTextEncoderConfig] = field(
        default_factory=lambda: dict(),
    )
