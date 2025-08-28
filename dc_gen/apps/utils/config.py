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

from typing import TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


def get_config(config_class: type[T]) -> T:
    cfg = OmegaConf.structured(config_class)

    additional_cfg = OmegaConf.from_cli()
    if "yaml" in additional_cfg:
        yaml_cfg = OmegaConf.load(additional_cfg.yaml)
        yaml_cfg = OmegaConf.masked_copy(yaml_cfg, cfg.keys())
        additional_cfg = OmegaConf.merge(yaml_cfg, additional_cfg)
        additional_cfg.pop("yaml")

    if "json" in additional_cfg:
        additional_cfg = OmegaConf.merge(additional_cfg.json, additional_cfg)
        additional_cfg.pop("json")

    cfg = OmegaConf.to_object(OmegaConf.merge(cfg, additional_cfg))
    return cfg
