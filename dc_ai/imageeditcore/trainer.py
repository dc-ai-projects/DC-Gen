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
from io import BytesIO
from typing import Any, Optional

os.environ["TOKENIZERS_PARALLELISM"] = (
    "true"  # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
)

import numpy as np
import pandas
import torch
from omegaconf import MISSING
from PIL import Image
from torch import nn
from tqdm import tqdm

from ..aecore.autoencoder import Autoencoder, AutoencoderConfig
from ..apps.metrics.clip_score import I2ICLIPScoreStats
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_barrier, is_dist_initialized, is_master, sync_tensor
from ..apps.utils.dtype import get_dtype_from_str
from ..apps.utils.metric import AverageMeter
from ..models.utils.network import freeze_weights, get_params_num
from .data_provider.base_eval import ImageEditCoreEvalDataProvider
from .data_provider.gedit import GEditDataProvider, GEditDataProviderConfig
from .data_provider.latent_mixture import (
    ImageEditCoreLatentMixtureDataProvider,
    ImageEditCoreLatentMixtureDataProviderConfig,
)
from .image_encoder import ImageEditCoreImageEncoder, ImageEditCoreImageEncoderConfig
from .models.base import BaseImageEditModel, BaseImageEditModelConfig
from .models.qwen_image import QwenImage, QwenImageConfig
from .text_encoder import ImageEditCoreTextEncoder

__all__ = ["ImageEditCoreTrainerConfig", "ImageEditCoreTrainer"]


@dataclass
class ImageEditCoreTrainerConfig(BaseTrainerConfig):
    # env
    resolution: str = "512F32"
    save_image_format: str = "jpg"
    cfg_scale: float = 4.0

    # text encoder
    text_encoders: tuple[str, ...] = ()
    use_neg_prompt: bool = True

    # image encoder
    image_encoder: ImageEditCoreImageEncoderConfig = field(default_factory=ImageEditCoreImageEncoderConfig)

    # eval data providers
    eval_data_providers: tuple[str, ...] = ()
    gedit: GEditDataProviderConfig = field(
        default_factory=lambda: GEditDataProviderConfig(resolution="${..resolution}")
    )
    max_eval_steps: int = 100000

    # train data providers
    train_data_scaling_factor: Any = None
    mixture: ImageEditCoreLatentMixtureDataProviderConfig = field(
        default_factory=lambda: ImageEditCoreLatentMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}"
        )
    )

    # autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    autoencoder_dtype: str = "fp32"

    # model
    model: str = MISSING
    model_dtype: Optional[str] = None
    qwen_image: QwenImageConfig = field(default_factory=lambda: QwenImageConfig(text_encoder_id="${..text_encoders.0}"))

    # metrics
    compute_clip_score: bool = True
    compute_metrics_with_jpeg: bool = False

    # dc-ae 1.5
    adaptive_latent_channels: Optional[tuple[int]] = None
    num_sample_latent_channels: int = 1
    always_sample_max_channel: bool = False

    # steps
    save_samples_steps: Optional[int] = 500
    save_checkpoint_steps: int = 500
    eval_steps: int = 5000

    # save format
    save_input_images: bool = False
    skip_exist: bool = False


class ImageEditCoreTrainer(BaseTrainer):
    def __init__(self, cfg: ImageEditCoreTrainerConfig):
        if cfg.resolution in ["512F32MS", "512F64MS"]:
            self.resolution = 512
        elif cfg.resolution == "1024":
            self.resolution = 1024
        elif cfg.resolution == "2048":
            self.resolution = 2048
        else:
            raise ValueError(f"Resolution {cfg.resolution} is not supported")

        super().__init__(cfg)
        self.cfg: ImageEditCoreTrainerConfig

    def build_eval_data_providers(self) -> list[ImageEditCoreEvalDataProvider]:
        eval_data_providers = []
        for eval_data_provider_name in self.cfg.eval_data_providers:
            if eval_data_provider_name == "GEdit":
                eval_data_providers.append(GEditDataProvider(self.cfg.gedit))
            else:
                raise ValueError(f"eval data provider {eval_data_provider_name} is not supported")
        return eval_data_providers

    def build_train_data_provider(self) -> ImageEditCoreLatentMixtureDataProvider:
        train_data_provider = ImageEditCoreLatentMixtureDataProvider(self.cfg.mixture)
        return train_data_provider

    def get_possible_models(self) -> dict[str, tuple[BaseImageEditModelConfig, type[BaseImageEditModel]]]:
        possible_models = {
            "qwen_image": (self.cfg.qwen_image, QwenImage),
        }
        return possible_models

    def build_model(self) -> BaseImageEditModel:
        possible_models = self.get_possible_models()

        if self.cfg.model in possible_models:
            model_cfg, model_class = possible_models[self.cfg.model]
            model_cfg.input_size = self.resolution // self.autoencoder.spatial_compression_ratio
            model = model_class(model_cfg)
        else:
            raise ValueError(f"model {self.cfg.model} is not supported among {possible_models.keys()}")

        if self.cfg.model_dtype is not None:
            model = model.to(dtype=get_dtype_from_str(self.cfg.model_dtype))

        if is_master():
            print(f"params: {get_params_num(model):.2f} M")

        return model

    def setup_model(self) -> None:
        self.text_encoder = ImageEditCoreTextEncoder(self.cfg.text_encoders).to(device=self.device)
        freeze_weights(self.text_encoder)
        self.image_encoder = ImageEditCoreImageEncoder(self.cfg.image_encoder).to(device=self.device)
        if self.cfg.model == "qwen_image":
            self.spatial_patch_size = self.cfg.qwen_image.patch_size
        else:
            raise NotImplementedError(f"Model {self.cfg.model} is not supported.")

        self.autoencoder = Autoencoder(self.cfg.autoencoder).to(
            device=self.device, dtype=get_dtype_from_str(self.cfg.autoencoder_dtype)
        )
        freeze_weights(self.autoencoder)
        super().setup_model()

    def get_train_data_provider_states(self, place_holder: bool = False) -> dict[str, Any]:
        train_data_provider_states = {}
        train_data_provider_states["sampler_state_dict"] = self.train_data_provider.sampler.state_dict(
            self.global_step, place_holder=place_holder
        )
        return train_data_provider_states

    def set_train_data_provider_states(self, train_data_provider_states: dict[str, Any]):
        self.train_data_provider.sampler.load_state_dict(train_data_provider_states["sampler_state_dict"])

    @torch.no_grad()
    def evaluate_single_image_edit(
        self,
        step: int,
        network: BaseImageEditModel,
        f_log=sys.stdout,
        cfg_scale: float = 4.0,
        additional_dir_name: str = "cfg_4.0",
    ) -> dict[str, Any]:
        network.eval()
        eval_generator = torch.Generator(device=torch.device("cuda"))
        eval_generator.manual_seed(self.cfg.seed + self.rank)

        data_provider = self.eval_data_providers[0]
        data_loader = data_provider.data_loader

        # metrics
        if self.cfg.compute_clip_score:
            clip_score_stats = I2ICLIPScoreStats()

        if self.cfg.eval_dir_name is not None:
            eval_dir = os.path.join(self.cfg.run_dir, self.cfg.eval_dir_name, additional_dir_name)
        else:
            eval_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(eval_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        def update_metrics(images1, images2):
            if self.cfg.compute_clip_score:
                clip_score_stats.update(images1, images2)

        idx = 0
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
                names = samples["name"]
                images = samples["image"]
                prompts = samples["prompt"]
                text_embed_info = self.text_encoder.get_text_embed_info(
                    prompts=prompts, images=images, device=self.device
                )
                if self.cfg.use_neg_prompt:
                    neg_text_embed_info = self.text_encoder.get_text_embed_info(
                        prompts=[" " for _ in range(len(prompts))],
                        images=images,
                        device=self.device,
                    )
                else:
                    neg_text_embed_info = None

                image_embed_info = self.image_encoder.get_image_embed_info(
                    images=images,
                    autoencoder=self.autoencoder,
                )

                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    latent_samples = network(
                        "generate",
                        text_embed_info=text_embed_info,
                        neg_text_embed_info=neg_text_embed_info,
                        image_embed_info=image_embed_info,
                        cfg_scale=cfg_scale,
                        generator=eval_generator,
                    )

                if self.cfg.autoencoder.name in ["dc-ae-f32c32-in-1.0"] and self.cfg.model in ["qwen_image"]:
                    latent_samples = latent_samples.squeeze(2)

                image_samples = self.autoencoder.decode(
                    latent_samples.to(get_dtype_from_str(self.cfg.autoencoder_dtype))
                )
                images_tensor = (image_samples * 0.5 + 0.5).clamp(0, 1)
                input_images_tensor = (images * 0.5 + 0.5).clamp(0, 1).to(images_tensor.device)
                width = images_tensor.shape[-1]
                images_tensor = torch.cat((input_images_tensor, images_tensor), dim=-1)
                image_samples_uint8 = torch.clamp(255.0 * images_tensor + 0.5, 0, 255).to(dtype=torch.uint8)
                images_numpy_uint8 = image_samples_uint8.cpu().permute(0, 2, 3, 1).numpy()

                if (
                    num_saved_samples < self.cfg.num_save_samples
                    and (is_master() or self.cfg.save_samples_at_all_ranks)
                ) or self.cfg.save_all_samples:
                    if self.cfg.save_input_images:
                        image_samples_PIL = [Image.fromarray(image) for image in images_numpy_uint8]
                    else:
                        image_samples_PIL = [Image.fromarray(image) for image in images_numpy_uint8[:, :, width:, :]]
                    for j, image_sample_PIL in enumerate(image_samples_PIL):
                        image_sample_PIL.save(
                            os.path.join(
                                eval_dir,
                                f"{names[j]}.{self.cfg.save_image_format}",
                            )
                        )
                        num_saved_samples += 1
                    del image_samples_PIL

                if self.cfg.compute_metrics_with_jpeg:
                    output_images_jpeg = []
                    for image in image_samples_uint8:
                        with BytesIO() as buff:
                            Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(buff, format="JPEG")
                            buff.seek(0)
                            out = buff.read()
                            output_images_jpeg.append(
                                torch.tensor(np.array(Image.open(BytesIO(out)))).permute(2, 0, 1).cuda()
                            )
                    image_samples_uint8 = torch.stack(output_images_jpeg)
                else:
                    image_samples_uint8 = images_numpy_uint8

                images_tensor_uint8 = torch.from_numpy(image_samples_uint8).permute(0, 3, 1, 2)
                update_metrics(images_tensor_uint8[:, :, :, :width], images_tensor_uint8[:, :, :, width:])

                # clip-score
                if self.cfg.compute_clip_score:
                    clip_res = clip_score_stats.compute()
                    if is_master():
                        print("CLIP Score:", clip_res)
                # tqdm
                t.update()

                idx += 1
                if idx >= self.cfg.max_eval_steps:
                    break

        eval_info_dict = dict()
        torch.cuda.empty_cache()

        # clip-score
        if self.cfg.compute_clip_score:
            eval_info_dict["clip_score"] = clip_score_stats.compute()
        return eval_info_dict

    def evaluate(self, step: int, model: BaseImageEditModel, f_log=sys.stdout) -> dict[Any, Any]:
        torch.cuda.empty_cache()
        self.autoencoder = self.autoencoder.to(self.device)
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
            eval_info_dict[setting] = {}
            if index in results.index:
                eval_info_dict[setting] = results.loc[[index]].to_dict(orient="index")[index]
            else:
                eval_info_dict[setting]["image_edit"] = self.evaluate_single_image_edit(
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
        self.autoencoder = self.autoencoder.to("cpu")
        return eval_info_dict

    def get_trainable_module_list(self, model: BaseImageEditModel) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().prepare_batch(batch)

        if self.cfg.adaptive_latent_channels is not None:
            if self.cfg.always_sample_max_channel:
                indices = torch.randint(
                    0,
                    len(self.cfg.adaptive_latent_channels),
                    (self.cfg.num_sample_latent_channels - 1,),
                    generator=self.train_async_generator_gpu,
                    device=self.device,
                )
                latent_channels = [self.cfg.adaptive_latent_channels[-1]] + [
                    self.cfg.adaptive_latent_channels[index.item()] for index in indices
                ]
            else:
                indices = torch.randint(
                    0,
                    len(self.cfg.adaptive_latent_channels),
                    (self.cfg.num_sample_latent_channels,),
                    generator=self.train_async_generator_gpu,
                    device=self.device,
                )
                latent_channels = [self.cfg.adaptive_latent_channels[index.item()] for index in indices]
            batch["latent_channels"] = latent_channels
        else:
            batch["latent_channels"] = None

        return batch

    def model_forward(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        with torch.no_grad():
            images = batch["raw_images"]
            prompts = batch["captions"]

            if self.cfg.model == "qwen_image":
                neg_prompts = [" " for _ in range(len(prompts))]
                drop_ids = torch.rand(len(prompts)).cuda() < self.cfg.qwen_image.drop_text_ratio
                drop_ids_list = drop_ids.cpu().tolist()
                real_prompts = [neg_prompts[i] if drop_ids_list[i] else prompts[i] for i in range(len(prompts))]
            else:
                raise ValueError(f"Model {self.cfg.model} is not supported")

            text_embed_info = self.text_encoder.get_text_embed_info(
                prompts=real_prompts, images=images, device=self.device
            )
            image_embed_info = {
                "ae_features": batch["ae_features"],
            }

        if self.cfg.adaptive_latent_channels is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                latent_channels = batch["latent_channels"]
                loss_dict_list, info_list = zip(
                    *[
                        self.model(
                            "forward_train",
                            x=batch["images"][:, :latent_channel],
                            text_embed_info=text_embed_info,
                            image_embed_info=image_embed_info,
                            generator=self.train_async_generator_gpu,
                        )
                        for latent_channel in latent_channels
                    ]
                )
                loss_dict = {
                    key: torch.mean(torch.stack([loss_dict_[key] for loss_dict_ in loss_dict_list]))
                    for key in loss_dict_list[0]
                }

                def combine_info_list(info_list):
                    info_0 = info_list[0]
                    if isinstance(info_0, torch.Tensor):
                        info = torch.stack([info_ for info_ in info_list]).mean(dim=0)
                    elif isinstance(info_0, dict):
                        info = {key: combine_info_list([info_[key] for info_ in info_list]) for key in info_0}
                    elif isinstance(info_0, float):
                        info = np.mean([info_ for info_ in info_list])
                    else:
                        raise ValueError(f"info type {type(info_0)} is not supported")
                    return info

                info = combine_info_list(info_list)
        else:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                loss_dict, info = self.model(
                    "forward_train",
                    x=batch["images"],
                    text_embed_info=text_embed_info,
                    image_embed_info=image_embed_info,
                    generator=self.train_async_generator_gpu,
                )

        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        for i in range(2):
            if self.rank == i:
                print(
                    f"global step {self.global_step}, rank {self.rank}, images {batch['images'].sum()}, loss {loss_dict[0].item()}, grad_norm {info['grad_norm_0']}, memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
            dist_barrier()

    def save_samples(self, batch: dict[str, Any], info: dict[str, Any]):
        pass

    def get_current_step_train_loss_dict(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
        return info["detailed_loss_dict"]

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        return batch["images"].shape[0]

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
        postfix_dict["shape"] = batch["images"].shape

        for i in range(len(self.optimizers)):
            if f"mean_loss_{i}" in info:
                postfix_dict[f"mean_loss_{i}"] = info[f"mean_loss_{i}"]
                log_dict[f"train/mean_loss_{i}"] = info[f"mean_loss_{i}"]

    def check_termination(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> bool:
        mean_loss = sync_tensor(loss_dict[0], reduce="mean").item()
        if np.isnan(mean_loss):
            self.print_and_train_log(f"NaN detected, terminate training")
            return True
        return super().check_termination(loss_dict, info)


def main():
    cfg: ImageEditCoreTrainerConfig = get_config(ImageEditCoreTrainerConfig)
    trainer = ImageEditCoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
