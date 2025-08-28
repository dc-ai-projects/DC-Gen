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

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from dc_gen.apps.metrics.cmmd.cmmd import CMMDStats, CMMDStatsConfig
from dc_gen.apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from dc_gen.apps.utils.dist import dist_init


@dataclass
class ProcessGuidedDiffusionReferenceConfig:
    input_path: str = "assets/data/precision_recall/VIRTUAL_imagenet256_labeled.npz"
    precision_recall_output_path: str = "assets/data/precision_recall/VIRTUAL_imagenet256.npy"
    cmmd_output_path: str = "assets/data/cmmd/VIRTUAL_imagenet256.npy"
    batch_size: int = 256


def main():
    dist_init()
    cfg: ProcessGuidedDiffusionReferenceConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(ProcessGuidedDiffusionReferenceConfig), OmegaConf.from_cli())
    )
    fid_stats = FIDStats(
        FIDStatsConfig(precision_recall_ref_path=cfg.input_path)
    )  # precision_recall_ref_path=cfg.input_path acts as a placeholder
    cmmd_stats = CMMDStats(CMMDStatsConfig())

    device = torch.device("cuda")
    arr_0 = np.load(cfg.input_path)["arr_0"]

    for start in tqdm(range(0, arr_0.shape[0], cfg.batch_size)):
        batch = torch.tensor(arr_0[start : min(start + cfg.batch_size, arr_0.shape[0])], device=device).permute(
            0, 3, 1, 2
        )
        fid_stats.add_data(batch)
        cmmd_stats.add_data(batch)

    pred_arr = np.concatenate(fid_stats.pred_arr)
    os.makedirs(os.path.dirname(cfg.precision_recall_output_path), exist_ok=True)
    np.save(cfg.precision_recall_output_path, pred_arr)

    embs = cmmd_stats.get_stats()
    os.makedirs(os.path.dirname(cfg.cmmd_output_path), exist_ok=True)
    np.save(cfg.cmmd_output_path, embs)


if __name__ == "__main__":
    main()

"""
python -m scripts.data.process_guided_diffusion_reference

python -m scripts.data.process_guided_diffusion_reference input_path=../metric/guided-diffusion/evaluations/VIRTUAL_imagenet512.npz precision_recall_output_path=assets/data/precision_recall/VIRTUAL_imagenet512.npy cmmd_output_path=assets/data/cmmd/VIRTUAL_imagenet512.npy
"""
