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

import pandas
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ...apps.utils.dist import dist_init, get_dist_local_rank, get_dist_size, is_dist_initialized, is_master
from .collection import possible_eval_data_providers, possible_train_data_providers


@dataclass
class VLMCaptionConfig:
    data_provider: str = "ImageNetEval1k"
    batch_size: int = 32
    vlm: str = "OpenGVLab/InternVL2-1B"


def main():
    dist_init()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(get_dist_local_rank())
    torch.set_grad_enabled(False)

    cfg: VLMCaptionConfig = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(VLMCaptionConfig), OmegaConf.from_cli())
    )

    if cfg.data_provider.endswith("Eval") and cfg.data_provider.removesuffix("Eval") in possible_eval_data_providers:
        data_provider_config_class, data_provider_class = possible_eval_data_providers[
            cfg.data_provider.removesuffix("Eval")
        ]
    elif (
        cfg.data_provider.endswith("Train") and cfg.data_provider.removesuffix("Train") in possible_train_data_providers
    ):
        data_provider_config_class, data_provider_class = possible_train_data_providers[
            cfg.data_provider.removesuffix("Train")
        ]
    else:
        raise ValueError(f"data_provider {cfg.data_provider} is not supported")
    assert data_provider_config_class().name == cfg.data_provider

    if cfg.vlm in ["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-26B"]:
        vlm = (
            AutoModel.from_pretrained(
                cfg.vlm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
            )
            .eval()
            .cuda()
        )
        tokenizer = AutoTokenizer.from_pretrained(cfg.vlm, trust_remote_code=True, use_fast=False)
        generation_config = dict(max_new_tokens=512, do_sample=False)
        question = "<image>\nPlease describe the picture in detail"

        data_provider = data_provider_class(
            data_provider_config_class(
                resolution=448,
                batch_size=cfg.batch_size,
                shuffle=False,
                drop_last=False,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    else:
        raise ValueError(f"vlm {cfg.vlm} is not supported")

    outputs = {}
    with tqdm(
        total=len(data_provider.data_loader),
        desc=f"Using {cfg.vlm} to caption {cfg.data_provider}",
        disable=not is_master(),
        mininterval=10.0,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    ) as t:
        for images, labels in data_provider.data_loader:
            if cfg.vlm in ["OpenGVLab/InternVL2-1B", "OpenGVLab/InternVL2-26B"]:
                images = images.to(torch.bfloat16).cuda()
                batch_outputs = vlm.batch_chat(
                    tokenizer,
                    images,
                    num_patches_list=[1] * images.shape[0],
                    questions=[question] * images.shape[0],
                    generation_config=generation_config,
                )
                for i, output in enumerate(batch_outputs):
                    outputs[labels["index"][i].item()] = {"caption": output.replace("\n", " ")}
            else:
                raise ValueError(f"vlm {cfg.vlm} is not supported")
            t.update()

    data_frame = pandas.DataFrame.from_dict(outputs, orient="index")
    if is_dist_initialized():
        data_frames = [None for _ in range(get_dist_size())]
        dist.all_gather_object(data_frames, data_frame)
        data_frame = pandas.concat(data_frames).sort_index()

    if is_master():
        output_path = os.path.join(
            "assets", "data", "vlm_caption", f"{cfg.vlm.replace('/', '_')}_{data_provider.cfg.name}.csv"
        )
        data_frame.to_csv(output_path)


if __name__ == "__main__":
    main()


"""
python -m dc_gen.aecore.data_provider.vlm_caption

torchrun --nnodes=1 --nproc_per_node=8 -m dc_gen.aecore.data_provider.vlm_caption vlm=OpenGVLab/InternVL2-26B data_provider=ImageNetEval
"""
