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
from io import BytesIO

import numpy as np
import torch
from omegaconf import MISSING
from PIL import Image
from tqdm import tqdm

from dc_gen.aecore.data_provider.imagenet import (
    ImageNetDataProvider,
    ImageNetEvalDataProviderConfig,
    ImageNetTrainDataProviderConfig,
)
from dc_gen.aecore.data_provider.mjhq import MJHQDataProvider, MJHQEvalDataProviderConfig
from dc_gen.apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from dc_gen.apps.utils.config import get_config
from dc_gen.apps.utils.dist import dist_init, get_dist_local_rank, is_master


@dataclass
class GenerateReferenceConfig:
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)

    # dataset
    dataset: str = MISSING
    imagenet_train: ImageNetTrainDataProviderConfig = field(default_factory=ImageNetTrainDataProviderConfig)
    imagenet_eval: ImageNetEvalDataProviderConfig = field(default_factory=ImageNetEvalDataProviderConfig)
    mjhq: MJHQEvalDataProviderConfig = field(default_factory=MJHQEvalDataProviderConfig)


def main():
    cfg = get_config(GenerateReferenceConfig)

    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())

    if cfg.dataset == "imagenet_eval":
        data_provider = ImageNetDataProvider(cfg.imagenet_eval)
    elif cfg.dataset == "imagenet_train":
        data_provider = ImageNetDataProvider(cfg.imagenet_train)
    elif cfg.dataset == "mjhq":
        data_provider = MJHQDataProvider(cfg.mjhq)
    else:
        raise NotImplementedError

    fid_stats = FIDStats(cfg.fid)

    def add_data_from_dataloader(data_loader):
        for images, _ in tqdm(data_loader, disable=not is_master()):
            if cfg.dataset in ["imagenet_resize_jpeg_eval"]:
                images_uint8 = (255 * images + 0.5).clamp(0, 255).to(torch.uint8)
                images_jpeg = []
                for image in images_uint8:
                    with BytesIO() as buff:
                        Image.fromarray(image.permute(1, 2, 0).numpy()).save(buff, format="JPEG")
                        buff.seek(0)
                        out = buff.read()
                        images_jpeg.append(torch.tensor(np.array(Image.open(BytesIO(out)))).permute(2, 0, 1))
                images = torch.stack(images_jpeg) / 255

            images = images.cuda()
            fid_stats.add_data(images)

    add_data_from_dataloader(data_provider.data_loader)
    fid_stats.get_stats()


if __name__ == "__main__":
    main()
