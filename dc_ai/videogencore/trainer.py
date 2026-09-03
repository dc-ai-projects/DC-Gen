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
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import imageio
import ipdb
import pandas
import torch
from omegaconf import MISSING
from PIL import Image
from torch import nn
from tqdm import tqdm

from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.trainer.fsdp_wrap import fsdp_wrap
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_barrier, is_dist_initialized, is_master
from ..apps.utils.dtype import get_dtype_from_str
from ..apps.utils.metric import AverageMeter
from ..apps.utils.video import write_video
from ..models.utils.network import freeze_weights, get_params_num
from ..staecore.autoencoder import Autoencoder, AutoencoderConfig
from .data_provider.base_eval import VideoGenCoreEvalDataProvider
from .data_provider.latent_mixture import (
    VideoGenCoreLatentMixtureDataProvider,
    VideoGenCoreLatentMixtureDataProviderConfig,
)
from .data_provider.vbench import (
    VBenchImagePromptDataProvider,
    VBenchImagePromptDataProviderConfig,
    VBenchTextPromptDataProvider,
    VBenchTextPromptDataProviderConfig,
)
from .data_provider.vbench2 import VBench2DataProvider, VBench2DataProviderConfig
from .image_encoder import ImageEncoder, ImageEncoderConfig
from .models.base import BaseVideoGenModel, BaseVideoGenModelConfig
from .models.moe_wan_i2v import MoEWanI2V, MoEWanI2VConfig
from .models.moe_wan_t2v import MoEWanT2V, MoEWanT2VConfig
from .models.wan_i2v import WanI2V, WanI2VConfig
from .models.wan_t2v import WanT2V, WanT2VConfig
from .text_encoder import VideoGenCoreTextEncoder

__all__ = ["VideoGenCoreTrainerConfig", "VideoGenCoreTrainer"]


@dataclass
class VideoGenCoreTrainerConfig(BaseTrainerConfig):
    # env
    cfg_scale: float = 5.0
    pag_scale: float = 1.0
    resolution: str = "480"  # resolution for evaluation
    image_embed_info_num_frames: Optional[int] = None

    # text encoder
    text_encoders: tuple[str, ...] = ()

    # eval data providers
    eval_data_providers: tuple[str, ...] = ("VBenchTextPrompt",)
    vbench_text_prompt: VBenchTextPromptDataProviderConfig = field(default_factory=VBenchTextPromptDataProviderConfig)
    vbench_image_prompt: VBenchImagePromptDataProviderConfig = field(
        default_factory=VBenchImagePromptDataProviderConfig
    )
    vbench2: VBench2DataProviderConfig = field(default_factory=VBench2DataProviderConfig)
    category_ids: tuple[int, ...] = (0, 3, 8, 9)
    skip_vbench_evaluator: bool = False

    # train data providers
    mixture: VideoGenCoreLatentMixtureDataProviderConfig = field(
        default_factory=lambda: VideoGenCoreLatentMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}"
        )
    )

    # autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    autoencoder_dtype: str = "fp32"

    # image_encoder
    image_encoder: ImageEncoderConfig = field(default_factory=lambda: ImageEncoderConfig(resolution="${..resolution}"))

    # model
    model: str = MISSING
    model_dtype: Optional[str] = None
    wan_t2v: WanT2VConfig = field(default_factory=lambda: WanT2VConfig(text_encoder_id="${..text_encoders.0}"))
    wan_i2v: WanI2VConfig = field(default_factory=lambda: WanI2VConfig(text_encoder_id="${..text_encoders.0}"))
    moe_wan_t2v: MoEWanT2VConfig = field(
        default_factory=lambda: MoEWanT2VConfig(text_encoder_id="${..text_encoders.0}", offload="${..offload}")
    )
    moe_wan_i2v: MoEWanI2VConfig = field(
        default_factory=lambda: MoEWanI2VConfig(text_encoder_id="${..text_encoders.0}", offload="${..offload}")
    )

    # lora
    use_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 128
    lora_alpha: int = 256

    # steps
    save_samples_steps: Optional[int] = 500
    save_checkpoint_steps: int = 500
    eval_steps: int = 5000

    # offload
    offload: bool = False


class VideoGenCoreTrainer(BaseTrainer):
    def __init__(self, cfg: VideoGenCoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: VideoGenCoreTrainerConfig

    def build_eval_data_providers(self) -> list[VideoGenCoreEvalDataProvider]:
        eval_data_providers = []
        for eval_data_provider_name in self.cfg.eval_data_providers:
            if eval_data_provider_name == "VBenchTextPrompt":
                eval_data_providers.append(VBenchTextPromptDataProvider(self.cfg.vbench_text_prompt))
            elif eval_data_provider_name == "VBenchImagePrompt":
                eval_data_providers.append(VBenchImagePromptDataProvider(self.cfg.vbench_image_prompt))
            elif eval_data_provider_name == "VBench2":
                eval_data_providers.append(VBench2DataProvider(self.cfg.vbench2))
            else:
                raise ValueError(f"eval data provider {eval_data_provider_name} is not supported")
        return eval_data_providers

    def build_train_data_provider(self) -> VideoGenCoreLatentMixtureDataProvider:
        train_data_provider = VideoGenCoreLatentMixtureDataProvider(self.cfg.mixture)
        return train_data_provider

    def get_possible_models(self) -> dict[str, tuple[BaseVideoGenModelConfig, type[BaseVideoGenModel]]]:
        possible_models = {
            "wan_t2v": (self.cfg.wan_t2v, WanT2V),
            "wan_i2v": (self.cfg.wan_i2v, WanI2V),
            "moe_wan_t2v": (self.cfg.moe_wan_t2v, MoEWanT2V),
            "moe_wan_i2v": (self.cfg.moe_wan_i2v, MoEWanI2V),
        }
        return possible_models

    def build_model(self) -> BaseVideoGenModel:
        possible_models = self.get_possible_models()

        if self.cfg.model in possible_models:
            model_cfg, model_class = possible_models[self.cfg.model]
            if self.cfg.model == "wan_i2v":
                self.cfg.wan_i2v.t_ratio = self.autoencoder.temporal_compression_ratio
            elif self.cfg.model == "moe_wan_i2v":
                self.cfg.moe_wan_i2v.t_ratio = self.autoencoder.temporal_compression_ratio
            elif self.cfg.model in [
                "wan_t2v",
                "moe_wan_t2v",
            ]:
                pass
            else:
                raise ValueError(f"Model type {self.cfg.model} is not supported.")
            model = model_class(model_cfg)
        else:
            raise ValueError(f"model {self.cfg.model} is not supported among {possible_models.keys()}")

        if self.cfg.model_dtype is not None:
            model = model.to(get_dtype_from_str(self.cfg.model_dtype))

        if is_master():
            print(f"params: {get_params_num(model):.2f} M")

        if self.cfg.mode == "eval":
            freeze_weights(model)  # Save Memory for T2V-14B, I2V-14B inference

        return model

    def setup_model(self) -> None:
        self.text_encoder = VideoGenCoreTextEncoder(self.cfg.text_encoders)
        freeze_weights(self.text_encoder)
        self.text_encoder.to("cpu" if self.cfg.offload else self.device)

        self.autoencoder = Autoencoder(self.cfg.autoencoder).to(dtype=get_dtype_from_str(self.cfg.autoencoder_dtype))
        freeze_weights(self.autoencoder)
        self.autoencoder.to("cpu" if self.cfg.offload else self.device)

        if self.cfg.model in ["wan_i2v", "moe_wan_i2v"]:
            self.image_encoder = ImageEncoder(self.cfg.image_encoder, self.device)
            freeze_weights(self.image_encoder.vlm.model)
            self.image_encoder.to("cpu" if self.cfg.offload else self.device)
            torch.cuda.empty_cache()
            if self.cfg.model == "wan_i2v":
                self.spatial_patch_size = self.cfg.wan_i2v.patch_size[1]
            elif self.cfg.model == "moe_wan_i2v":
                self.spatial_patch_size = self.cfg.moe_wan_i2v.patch_size[1]
        elif self.cfg.model in ["wan_t2v", "moe_wan_t2v"]:
            self.image_encoder = None
            if self.cfg.model == "wan_t2v":
                self.spatial_patch_size = self.cfg.wan_t2v.patch_size[1]
            elif self.cfg.model == "moe_wan_t2v":
                self.spatial_patch_size = self.cfg.moe_wan_t2v.patch_size[1]
        else:
            raise ValueError(f"model {self.cfg.model} is not supported")

        self.model = self.build_model()
        self.model.to("cpu" if self.cfg.offload else self.device)

        if self.cfg.use_lora or self.cfg.use_dora:
            self.model.lora_wrap(self.cfg.lora_rank, self.cfg.lora_alpha, self.cfg.use_lora, self.cfg.use_dora)

        torch.cuda.empty_cache()

    def setup_model_for_training(self):
        super().setup_model_for_training()
        # Prepare to shard text encoder (umt5-xxl)

        assert self.cfg.distributed_method in ["FSDP", "FSDPWrap"], "Must Use FSDP for Wan-Video"

        self.text_encoder = fsdp_wrap(self.text_encoder)

        torch.cuda.empty_cache()
        dist_barrier()

    @torch.no_grad()
    def generate(
        self,
        network: BaseVideoGenModel,
        prompt_list: list[str],
        image_list: Optional[list[Image.Image]],
        cfg_scale: float,
        pag_scale: float,
        eval_generator: torch.Generator,
    ):
        if self.cfg.offload:
            if self.cfg.model in [
                "wan_t2v",
                "wan_i2v",
            ]:
                network = network.to("cpu")
            elif self.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
                network.submodel[0] = network.submodel[0].to("cpu")
            else:
                raise ValueError(f"Model type {self.cfg.model} is not supported.")

            torch.cuda.empty_cache()

        if image_list is not None and self.image_encoder is not None:
            if self.cfg.offload:
                self.autoencoder.to(self.device)
                self.image_encoder.to(self.device)
            image_embed_info = self.image_encoder.get_image_embed_info(
                image_list, self.autoencoder, self.spatial_patch_size, self.cfg.image_embed_info_num_frames
            )
            if self.cfg.offload:
                self.image_encoder.to("cpu")
                self.autoencoder.to("cpu")
        else:
            image_embed_info = {}

        if self.cfg.offload:
            self.text_encoder.to(self.device)

        text_embed_info = self.text_encoder(prompt_list, self.device)

        if self.cfg.offload:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if self.cfg.model in [
                "wan_t2v",
                "wan_i2v",
            ]:
                network = network.to(self.device)
            elif self.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
                network.submodel[0] = network.submodel[0].to(self.device)
            else:
                raise ValueError(f"Model type {self.cfg.model} is not supported.")

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            latent_samples, _ = network(
                "generate",
                text_embed_info=text_embed_info,
                image_embed_info=image_embed_info,
                cfg_scale=cfg_scale,
                pag_scale=pag_scale,
                generator=eval_generator,
            )  # ([B,C,T,H,W], info)

        if self.cfg.offload:
            if self.cfg.model in [
                "wan_t2v",
                "wan_i2v",
            ]:
                network = network.to("cpu")
            elif self.cfg.model in ["moe_wan_t2v", "moe_wan_i2v"]:
                network.submodel[-1] = network.submodel[-1].to("cpu")
            else:
                raise ValueError(f"Model type {self.cfg.model} is not supported.")
            torch.cuda.empty_cache()
            self.autoencoder.to(self.device)

        video_samples = self.autoencoder.decode(latent_samples.to(get_dtype_from_str(self.cfg.autoencoder_dtype)))

        if self.cfg.offload:
            self.autoencoder.to("cpu")
            if self.cfg.model in [
                "wan_t2v",
                "wan_i2v",
            ]:
                network = network.to(self.device)
            else:
                raise ValueError(f"Model type {self.cfg.model} is not supported.")
            self.text_encoder.to(self.device)

        return video_samples

    def get_train_data_provider_states(self, place_holder: bool = False) -> dict[str, Any]:
        train_data_provider_states = {}
        train_data_provider_states["sampler_state_dict"] = self.train_data_provider.sampler.state_dict(
            self.global_step, place_holder=place_holder
        )
        return train_data_provider_states

    def set_train_data_provider_states(self, train_data_provider_states: dict[str, Any]):
        self.train_data_provider.sampler.load_state_dict(train_data_provider_states["sampler_state_dict"])

    @torch.no_grad()
    def evaluate_single(
        self,
        step: int,
        network: BaseVideoGenModel,
        f_log=sys.stdout,
        cfg_scale: float = 5.0,
        pag_scale: float = 1.0,
        additional_dir_name: Optional[str] = None,
    ) -> dict[str, Any]:
        device = torch.device("cuda")

        network.eval()
        eval_generator = torch.Generator(device=device)
        eval_generator.manual_seed(self.cfg.seed + self.rank)

        data_provider = self.eval_data_providers[0]
        data_loader = data_provider.data_loader

        if self.cfg.eval_dir_name is not None:
            eval_dir = os.path.join(self.cfg.run_dir, self.cfg.eval_dir_name, additional_dir_name)
        else:
            eval_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(eval_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()
        with tqdm(
            total=len(data_loader),
            desc="Valid Step #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            num_saved_samples = 0
            for _, samples in enumerate(data_loader):
                prompt_list = samples["extended_prompt"] if "extended_prompt" in samples else samples["prompt"]
                if self.cfg.model in [
                    "wan_i2v",
                    "moe_wan_i2v",
                ]:
                    image_path_list = samples["image_path"]
                    image_list = [Image.open(image_path) for image_path in image_path_list]
                elif self.cfg.model in [
                    "wan_t2v",
                    "moe_wan_t2v",
                ]:
                    image_list = None
                else:
                    raise ValueError(f"model {self.cfg.model} is not supported")

                video_samples = self.generate(network, prompt_list, image_list, cfg_scale, pag_scale, eval_generator)

                # assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                if isinstance(video_samples, list):
                    video_samples = torch.stack(video_samples, dim=0)  # [3,F,H,W]*B -> [B,3,F,H,W]
                video_samples_uint8 = torch.clamp(127.5 * video_samples + 127.5, 0, 255).to(dtype=torch.uint8)
                video_samples_numpy = video_samples_uint8.permute(0, 2, 3, 4, 1).cpu().numpy()

                if (
                    num_saved_samples < self.cfg.num_save_samples
                    and (is_master() or self.cfg.save_samples_at_all_ranks)
                ) or self.cfg.save_all_samples:
                    for j, path in enumerate(samples["path"]):
                        save_path = os.path.join(eval_dir, path)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        write_video(save_path, video_samples[j] * 0.5 + 0.5, fps=16)

                t.update()

        torch.cuda.empty_cache()

        if self.cfg.skip_vbench_evaluator:
            return {}

        if self.cfg.eval_data_providers[0] == "VBenchTextPrompt":
            batch_size, num_samples, num_videos_per_prompt = (
                self.cfg.vbench_text_prompt.batch_size,
                self.cfg.vbench_text_prompt.num_samples,
                self.cfg.vbench_text_prompt.num_videos_per_prompt,
            )
        elif self.cfg.eval_data_providers[0] == "VBenchImagePrompt":
            batch_size, num_samples, num_videos_per_prompt = (
                self.cfg.vbench_image_prompt.batch_size,
                self.cfg.vbench_image_prompt.num_samples,
                self.cfg.vbench_image_prompt.num_videos_per_prompt,
            )
        else:
            raise NotImplementedError(f"Vbench task {self.cfg.eval_data_providers[0]} is not supported")

        from .vbench_evaluator import VBenchEvaluator

        evaluator = VBenchEvaluator(
            self,
            network,
            batch_size,
            num_samples,
            num_videos_per_prompt,
            cfg_scale,
            pag_scale,
            eval_generator,
        )
        eval_info_dict = evaluator.evaluate(step, f_log)

        return eval_info_dict

    def evaluate(self, step: int, model: BaseVideoGenModel, f_log=sys.stdout) -> dict[Any, Any]:
        results_path = os.path.join(self.cfg.run_dir, "eval_results.csv")
        if os.path.exists(results_path):
            results = pandas.read_csv(results_path, index_col=0)
        else:
            results = pandas.DataFrame()
        eval_info_dict = {}
        cfg_scale_list = self.cfg.cfg_scale if isinstance(self.cfg.cfg_scale, list) else [self.cfg.cfg_scale]
        for cfg_scale in cfg_scale_list:
            eval_dir_name = f"{self.cfg.eval_dir_name}" if self.cfg.eval_dir_name is not None else f"step_{step}"
            setting = f"cfg_{cfg_scale}"
            index = f"{eval_dir_name}_{setting}"
            if index in results.index:
                eval_info_dict[setting] = results.loc[[index]].to_dict(orient="index")[index]
            else:
                eval_info_dict[setting] = self.evaluate_single(
                    step,
                    model,
                    f_log,
                    cfg_scale,
                    additional_dir_name=setting,
                )
                if os.path.exists(results_path):
                    results = pandas.read_csv(results_path, index_col=0)
                dist_barrier()
                if is_master():
                    results = pandas.concat(
                        [results, pandas.DataFrame.from_dict({index: eval_info_dict[setting]}, orient="index")]
                    ).sort_index()
                    results.to_csv(results_path)
        return eval_info_dict

    def get_trainable_module_list(self, model: BaseVideoGenModel) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def torch_compile(self):
        if self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            self.model.module.forward_train = torch.compile(self.model.module.forward_train)
        else:
            raise ValueError(f"Distributed method {self.cfg.distributed_method} not supported")

    def model_forward(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        with torch.no_grad():
            prompts = batch["captions"]
            text_embed_info = self.text_encoder(prompts=prompts, device=self.device)
            if self.cfg.model in [
                "wan_t2v",
                "moe_wan_t2v",
            ]:
                image_embed_info = {}
            elif self.cfg.model in [
                "wan_i2v",
                "moe_wan_i2v",
            ]:
                assert batch["videos"].shape[2] == batch["ae_feature"].shape[2]
                ae_masks = self.image_encoder.get_image_mask(batch["ae_feature"], self.autoencoder)
                image_embed_info = {
                    "ae_feature": batch["ae_feature"],
                    "vlm_feature": batch["vlm_feature"],
                    "img_masks": ae_masks,
                }
            else:
                raise ValueError(f"model {self.cfg.model} is not supported")

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            loss_dict, info = self.model(
                "forward_train",
                x=batch["videos"],
                text_embed_info=text_embed_info,
                image_embed_info=image_embed_info,
            )
        if "detailed_loss_dict" in info:
            for key in info["detailed_loss_dict"]:
                info["detailed_loss_dict"][key] = info["detailed_loss_dict"][key].item()

        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        for i in range(2):
            if self.rank == i:
                print(
                    f"global step {self.global_step}, rank {self.rank}, videos {batch['videos'].sum()}, loss {loss_dict[0].item()}, grad_norm {info['grad_norm_0']}",
                    flush=True,
                )
            dist_barrier()

    def save_samples(self, batch: dict[str, Any], info: dict[str, Any]):
        pass

    def get_current_step_train_loss_dict(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
        return info["detailed_loss_dict"]

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        return batch["videos"].shape[0]

    def after_step(
        self,
        batch: dict[str, Any],
        loss_dict: dict[str, Any],
        info: dict[str, Any],
        average_loss_dict: dict[str, AverageMeter],
        log_dict: dict[str, Any],
        postfix_dict: dict[str, Any],
    ) -> None:
        super().after_step(batch, loss_dict, info, average_loss_dict, log_dict, postfix_dict)
        postfix_dict["shape"] = batch["videos"].shape

        for i in range(len(self.optimizers)):
            if f"mean_loss_{i}" in info:
                postfix_dict[f"mean_loss_{i}"] = info[f"mean_loss_{i}"]
                log_dict[f"train/mean_loss_{i}"] = info[f"mean_loss_{i}"]

    def __del__(self):
        if is_master() and self.cfg.verbose:
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        super().__del__()


def main():
    cfg: VideoGenCoreTrainerConfig = get_config(VideoGenCoreTrainerConfig)
    trainer = VideoGenCoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
