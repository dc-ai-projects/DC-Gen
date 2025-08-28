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

import torch

from dc_gen.apps.utils.config import get_config
from dc_gen.t2icore.text_encoder import SingleTextEncoder, SingleTextEncoderConfig


def main():
    cfg = get_config(SingleTextEncoderConfig)
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    text_encoder = SingleTextEncoder(cfg).to(device)
    if cfg.name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        null_text_embeddings = text_encoder.get_null_embeddings(device, require_state_type="hidden_states")
    else:
        null_text_embeddings = text_encoder.get_null_embeddings(device)

    save_root = "assets/data/null_text_embeddings"
    os.makedirs(save_root, exist_ok=True)
    save_name = cfg.name.replace("/", "_").replace("-", "_") + ".pth"
    torch.save(null_text_embeddings.squeeze(0), os.path.join(save_root, save_name))


if __name__ == "__main__":
    main()
