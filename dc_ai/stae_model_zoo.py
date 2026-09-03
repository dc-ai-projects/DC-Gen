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

from typing import Callable, Optional

from .staecore.models.dc_ae_v import (
    DCAEVConfig,
    dc_ae_v_f32t4_chunk_causal,
    dc_ae_v_f64t4_chunk_causal,
)

REGISTERED_DCAEV_MODEL: dict[str, tuple[Callable[[str, str], DCAEVConfig], Optional[str]]] = {
    "dc-ae-v-1.0-f32t4c32": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c32.pt",
    ),
    "dc-ae-v-1.0-f32t4c64": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c64.pt",
    ),
    "dc-ae-v-1.0-f32t4c128": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c128.pt",
    ),
    "dc-ae-v-1.0-f32t4c256": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c256.pt",
    ),
    "dc-ae-v-1.0-f64t4c128": (
        dc_ae_v_f64t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f64t4c128.pt",
    ),
    "dc-ae-v-1.0-f32t4c32-bf16": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c32-bf16.pt",
    ),
    "dc-ae-v-1.0-f32t4c64-bf16": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c64-bf16.pt",
    ),
    "dc-ae-v-1.0-f32t4c128-bf16": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c128-bf16.pt",
    ),
    "dc-ae-v-1.0-f32t4c256-bf16": (
        dc_ae_v_f32t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f32t4c256-bf16.pt",
    ),
    "dc-ae-v-1.0-f64t4c128-bf16": (
        dc_ae_v_f64t4_chunk_causal,
        "assets/checkpoints/dc_videogen/dc_ae_v/dc-ae-v-1.0-f64t4c128-bf16.pt",
    ),
    #################################################################################################
}
