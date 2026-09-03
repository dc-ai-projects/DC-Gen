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

import sys
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..aecore.autoencoder import Autoencoder, AutoencoderConfig
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.utils.config import get_config
from ..apps.utils.dist import is_master
from ..apps.utils.dtype import get_dtype_from_str
from ..apps.utils.metric import AverageMeter
from ..c2icore.diffusioncore.models.dit import DiTConfig
from ..c2icore.diffusioncore.models.uvit import UViTConfig
from ..imageeditcore.models.qwen_image import QwenImageConfig
from ..models.utils import to_inference_mode
from ..staecore.autoencoder import Autoencoder as STAEAutoencoder
from ..staecore.autoencoder import AutoencoderConfig as STAEAutoencoderConfig
from ..t2icore.data_provider.image_mixture import (
    T2ICoreImageMixtureDataProvider,
    T2ICoreImageMixtureDataProviderConfig,
)
from ..t2icore.models.flux import FluxConfig
from ..t2icore.models.sana_t2i import SanaT2IConfig
from ..t2icore.models.zimage import ZImageConfig
from ..videogencore.image_encoder import ImageEncoder, ImageEncoderConfig
from .data_provider.base import DCAdaptLatentDataProvider
from .data_provider.imagenet import (
    DCAdaptImageNetImageProvider,
    DCAdaptImageNetImageProviderConfig,
    DCAdaptImageNetLatentProvider,
    DCAdaptImageNetLatentProviderConfig,
)
from .data_provider.latent_mixture import (
    DCAdaptLatentImageEditMixtureDataProviderConfig,
    DCAdaptLatentImageMixtureDataProviderConfig,
    DCAdaptLatentMixtureDataProvider,
    DCAdaptLatentVideoMixtureDataProviderConfig,
)
from .models.base import BasePatchEmbedding
from .models.dit import DiTPatchEmbedding
from .models.flux import FluxPatchEmbedding
from .models.qwen_image import QwenImagePatchEmbedding
from .models.sana_t2i import SanaT2IPatchEmbedding
from .models.uvit import UViTPatchEmbedding
from .models.wan_video import (
    WanI2VPatchEmbedding,
    WanI2VPatchEmbeddingConfig,
    WanT2VPatchEmbedding,
    WanT2VPatchEmbeddingConfig,
)
from .models.zimage import ZImagePatchEmbedding

__all__ = ["DCAdaptTrainerConfig", "DCAdaptTrainer"]


@dataclass
class DCAdaptTrainerConfig(BaseTrainerConfig):
    # env
    align_type: str = "upsample"
    upsample_mode: str = "bilinear"

    # eval
    eval_max_batch_num: int = 100

    # autoencoder
    teacher_ae: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    student_ae: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    teacher_autoencoder_dtype: str = "fp32"
    student_autoencoder_dtype: str = "fp32"
    teacher_stae: STAEAutoencoderConfig = field(default_factory=STAEAutoencoderConfig)
    student_stae: STAEAutoencoderConfig = field(default_factory=STAEAutoencoderConfig)

    # model
    model_type: str = "dit"

    teacher_dit: DiTConfig = field(default_factory=DiTConfig)
    student_dit: DiTConfig = field(default_factory=DiTConfig)
    teacher_uvit: UViTConfig = field(default_factory=UViTConfig)
    student_uvit: UViTConfig = field(default_factory=UViTConfig)
    teacher_sana_t2i: SanaT2IConfig = field(default_factory=SanaT2IConfig)
    student_sana_t2i: SanaT2IConfig = field(default_factory=SanaT2IConfig)
    teacher_flux: FluxConfig = field(default_factory=FluxConfig)
    student_flux: FluxConfig = field(default_factory=FluxConfig)
    teacher_zimage: ZImageConfig = field(default_factory=ZImageConfig)
    student_zimage: ZImageConfig = field(default_factory=ZImageConfig)
    teacher_qwen_image: QwenImageConfig = field(default_factory=QwenImageConfig)
    student_qwen_image: QwenImageConfig = field(default_factory=QwenImageConfig)

    teacher_wan_t2v: WanT2VPatchEmbeddingConfig = field(default_factory=WanT2VPatchEmbeddingConfig)
    student_wan_t2v: WanT2VPatchEmbeddingConfig = field(default_factory=WanT2VPatchEmbeddingConfig)
    teacher_wan_i2v: WanI2VPatchEmbeddingConfig = field(default_factory=WanI2VPatchEmbeddingConfig)
    student_wan_i2v: WanI2VPatchEmbeddingConfig = field(default_factory=WanI2VPatchEmbeddingConfig)

    # data_provider
    data_type: str = "latent"
    dataset: str = "imagenet"
    latent_imagenet: DCAdaptImageNetLatentProviderConfig = field(default_factory=DCAdaptImageNetLatentProviderConfig)
    image_imagenet: DCAdaptImageNetImageProviderConfig = field(default_factory=DCAdaptImageNetImageProviderConfig)
    latent_image: DCAdaptLatentImageMixtureDataProviderConfig = field(
        default_factory=lambda: DCAdaptLatentImageMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}",
        )
    )
    image_mixture: T2ICoreImageMixtureDataProviderConfig = field(
        default_factory=lambda: T2ICoreImageMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}",
        )
    )
    latent_image_edit: DCAdaptLatentImageEditMixtureDataProviderConfig = field(
        default_factory=lambda: DCAdaptLatentImageEditMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}",
        )
    )
    latent_video: DCAdaptLatentVideoMixtureDataProviderConfig = field(
        default_factory=lambda: DCAdaptLatentVideoMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}",
        )
    )
    teacher_latent_temporal_left_pad: int = 0
    teacher_latent_temporal_right_trim: int = 0

    # dc-ae 1.5
    adaptive_latent_channels: Optional[tuple[int]] = None
    num_sample_latent_channels: int = 1
    always_sample_max_channel: bool = False

    # image_encoder
    image_encoder: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)

    save_samples_steps: Optional[int] = None


class DCAdaptTrainer(BaseTrainer):
    def __init__(self, cfg: DCAdaptTrainerConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptTrainerConfig = cfg

    def build_eval_data_providers(self) -> list[DCAdaptLatentDataProvider]:
        return [self.build_train_data_provider()]

    def build_train_data_provider(self) -> DCAdaptLatentDataProvider:
        if self.cfg.data_type == "image":
            if self.cfg.dataset == "imagenet":
                train_data_provider = DCAdaptImageNetImageProvider(self.cfg.image_imagenet)
            elif self.cfg.dataset == "image_mixture":
                train_data_provider = T2ICoreImageMixtureDataProvider(self.cfg.image_mixture)
            else:
                raise NotImplementedError(f"Dataset {self.cfg.dataset} is not supported.")
        elif self.cfg.data_type == "latent":
            if self.cfg.dataset == "imagenet":
                train_data_provider = DCAdaptImageNetLatentProvider(self.cfg.latent_imagenet)
            elif self.cfg.dataset == "latent_image":
                train_data_provider = DCAdaptLatentMixtureDataProvider(self.cfg.latent_image)
            elif self.cfg.dataset == "latent_image_edit":
                train_data_provider = DCAdaptLatentMixtureDataProvider(self.cfg.latent_image_edit)
            elif self.cfg.dataset == "latent_video":
                train_data_provider = DCAdaptLatentMixtureDataProvider(self.cfg.latent_video)
            else:
                raise NotImplementedError(f"Dataset {self.cfg.dataset} is not supported.")
        else:
            raise NotImplementedError(f"Dataset Type {self.cfg.data_type} is not supported.")

        return train_data_provider

    def build_model(self) -> BasePatchEmbedding:
        if self.cfg.model_type == "dit":
            model = DiTPatchEmbedding(self.cfg.student_dit)
        elif self.cfg.model_type == "uvit":
            model = UViTPatchEmbedding(self.cfg.student_uvit)
            # load time token & label token pos_embed from teacher model
            model.pos_embed[:, :2, :].data.copy_(self.teacher_model.pos_embed[:, :2, :])
        elif self.cfg.model_type == "sana_t2i":
            model = SanaT2IPatchEmbedding(self.cfg.student_sana_t2i)
        elif self.cfg.model_type == "flux":
            model = FluxPatchEmbedding(self.cfg.student_flux)
        elif self.cfg.model_type == "zimage":
            model = ZImagePatchEmbedding(self.cfg.student_zimage)
        elif self.cfg.model_type == "qwen_image":
            model = QwenImagePatchEmbedding(self.cfg.student_qwen_image)
        elif self.cfg.model_type == "wan_t2v":
            model = WanT2VPatchEmbedding(self.cfg.student_wan_t2v)
        elif self.cfg.model_type == "wan_i2v":
            model = WanI2VPatchEmbedding(self.cfg.student_wan_i2v)
        else:
            raise NotImplementedError(f"Unsupported Model Type {self.cfg.model_type}")
        return model

    def setup_model(self) -> None:
        # autoencoder
        if self.cfg.data_type == "image":
            self.teacher_ae = Autoencoder(self.cfg.teacher_ae).to(
                device=self.device,
                dtype=get_dtype_from_str(self.cfg.teacher_autoencoder_dtype),
            )
            self.student_ae = Autoencoder(self.cfg.student_ae).to(
                device=self.device,
                dtype=get_dtype_from_str(self.cfg.student_autoencoder_dtype),
            )
            self.teacher_ae = to_inference_mode(self.teacher_ae)
            self.student_ae = to_inference_mode(self.student_ae)
        elif self.cfg.dataset in ["latent_image", "latent_image_edit"]:
            pass
        elif self.cfg.dataset in ["latent_video"]:
            # Load to get image mask
            self.teacher_ae = STAEAutoencoder(self.cfg.teacher_stae).to(device=self.device, dtype=torch.bfloat16)
            self.student_ae = STAEAutoencoder(self.cfg.student_stae).to(device=self.device, dtype=torch.bfloat16)
            self.teacher_ae = to_inference_mode(self.teacher_ae)
            self.student_ae = to_inference_mode(self.student_ae)
        else:
            raise ValueError(f"dataset {self.cfg.dataset} is not supported")

        # patch
        if self.cfg.model_type == "dit":
            teacher_model = DiTPatchEmbedding(self.cfg.teacher_dit)
        elif self.cfg.model_type == "uvit":
            teacher_model = UViTPatchEmbedding(self.cfg.teacher_uvit)
        elif self.cfg.model_type == "sana_t2i":
            teacher_model = SanaT2IPatchEmbedding(self.cfg.teacher_sana_t2i)
        elif self.cfg.model_type == "flux":
            teacher_model = FluxPatchEmbedding(self.cfg.teacher_flux)
        elif self.cfg.model_type == "zimage":
            teacher_model = ZImagePatchEmbedding(self.cfg.teacher_zimage)
        elif self.cfg.model_type == "qwen_image":
            teacher_model = QwenImagePatchEmbedding(self.cfg.teacher_qwen_image)
        elif self.cfg.model_type == "wan_t2v":
            teacher_model = WanT2VPatchEmbedding(self.cfg.teacher_wan_t2v)
        elif self.cfg.model_type == "wan_i2v":
            teacher_model = WanI2VPatchEmbedding(self.cfg.teacher_wan_i2v)
        else:
            raise NotImplementedError(f"Unsupported Teacher Model Architecture")
        self.teacher_model = to_inference_mode(teacher_model).to(self.device)

        if self.cfg.model_type == "wan_i2v":
            self.image_encoder = ImageEncoder(self.cfg.image_encoder, self.device)

        super().setup_model()

    def get_trainable_module_list(self, model: BasePatchEmbedding) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def forward_with_latents(
        self, teacher_latent: torch.Tensor, student_latent: torch.Tensor, model: Optional[nn.Module] = None
    ) -> dict[str, torch.Tensor]:
        model = self.model if model is None else model
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            with torch.no_grad():
                teacher_emb = self.teacher_model(teacher_latent).detach()
            student_emb = model(student_latent)

        # LTX Video
        if len(teacher_emb.shape) == 5:
            if self.cfg.student_stae.name == "Lightricks/LTX-Video-0.9.7-dev":
                student_emb = student_emb[:, :, :-1]
            if teacher_emb.shape[2] != student_emb.shape[2]:
                assert (
                    teacher_emb.shape[2] % student_emb.shape[2] == 0 or student_emb.shape[2] % teacher_emb.shape[2] == 0
                )
                temporal_downsample_coef = max(teacher_emb.shape[2], student_emb.shape[2]) // min(
                    teacher_emb.shape[2], student_emb.shape[2]
                )
                temporal_pool_layer = torch.nn.AvgPool3d(
                    kernel_size=(temporal_downsample_coef, 1, 1),
                    stride=(temporal_downsample_coef, 1, 1),
                )
                if teacher_emb.shape[2] > student_emb.shape[2]:
                    teacher_emb = temporal_pool_layer(teacher_emb)
                else:
                    student_emb = temporal_pool_layer(student_emb)

        tH, tW = teacher_emb.shape[-2], teacher_emb.shape[-1]
        sH, sW = student_emb.shape[-2], student_emb.shape[-1]
        if tH >= sH:
            assert tH % sH == 0 and tW % sW == 0 and tH // sH == tW // sW
            scale_factor = tH // sH
        else:
            assert sH % tH == 0 and sW % tW == 0 and sH // tH == sW // tW
            scale_factor = sH // tH

        # Average Pool Loss
        avgpool_loss = F.mse_loss(teacher_emb.mean(dim=(-2, -1)), student_emb.mean(dim=(-2, -1)))

        # Downsample Loss
        if len(teacher_emb.shape) == 4:
            pool_layer = torch.nn.AvgPool2d(kernel_size=scale_factor, stride=scale_factor)
        elif len(teacher_emb.shape) == 5:
            pool_layer = torch.nn.AvgPool3d(
                kernel_size=(1, scale_factor, scale_factor),
                stride=(1, scale_factor, scale_factor),
            )
        else:
            raise ValueError(f"teacher_emb shape {teacher_emb.shape} not supported")
        if tH >= sH:
            teacher_emb_down = pool_layer(teacher_emb)
            downsample_loss = F.mse_loss(teacher_emb_down, student_emb)
        else:
            student_emb_down = pool_layer(student_emb)
            downsample_loss = F.mse_loss(teacher_emb, student_emb_down)

        # Upsample Loss
        upsample_loss = {}

        if len(teacher_emb.shape) == 4:
            for upsample_mode in ["nearest", "bilinear", "bicubic"]:
                if tH >= sH:
                    student_emb_up = F.interpolate(student_emb, scale_factor=scale_factor, mode=upsample_mode)
                    upsample_loss[f"upsample_loss_{upsample_mode}"] = F.mse_loss(teacher_emb, student_emb_up)
                else:
                    teacher_emb_up = F.interpolate(teacher_emb, scale_factor=scale_factor, mode=upsample_mode)
                    upsample_loss[f"upsample_loss_{upsample_mode}"] = F.mse_loss(teacher_emb_up, student_emb)
        elif len(teacher_emb.shape) == 5:
            for upsample_mode in ["nearest", "trilinear"]:
                if tH >= sH:
                    student_emb_up = F.interpolate(
                        student_emb, scale_factor=(1, scale_factor, scale_factor), mode=upsample_mode
                    )
                    upsample_loss[f"upsample_loss_{upsample_mode}"] = F.mse_loss(teacher_emb, student_emb_up)
                else:
                    teacher_emb_up = F.interpolate(
                        teacher_emb, scale_factor=(1, scale_factor, scale_factor), mode=upsample_mode
                    )
                    upsample_loss[f"upsample_loss_{upsample_mode}"] = F.mse_loss(teacher_emb_up, student_emb)
        else:
            raise ValueError(f"teacher_emb shape {teacher_emb.shape} not supported")

        return {
            "avgpool_loss": avgpool_loss,
            "downsample_loss": downsample_loss,
            **upsample_loss,
        }

    def forward_with_images(self, images: torch.Tensor, model: Optional[nn.Module] = None) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_latent = self.teacher_ae.encode(
                images.to(dtype=get_dtype_from_str(self.cfg.teacher_autoencoder_dtype))
            ).detach()
            student_latent = self.student_ae.encode(
                images.to(dtype=get_dtype_from_str(self.cfg.student_autoencoder_dtype))
            ).detach()
        return self.forward_with_latents(teacher_latent, student_latent, model)

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

    def model_forward(
        self, batch: dict[str, Any], model: Optional[nn.Module] = None, training: bool = True
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:

        if self.cfg.adaptive_latent_channels is not None and training:
            assert self.cfg.data_type == "latent", "Adaptive Patch Alignment only support latent adaptation."
            latent_channels = batch["latent_channels"]
            teacher_latent = batch["data1"].cuda()
            student_latent = batch["data2"].cuda()
            loss_dict_list = [
                self.forward_with_latents(
                    teacher_latent,
                    student_latent[:, :latent_channel],
                    model,
                )
                for latent_channel in latent_channels
            ]
            loss_dict = {
                key: torch.mean(torch.stack([loss_dict_[key] for loss_dict_ in loss_dict_list]))
                for key in loss_dict_list[0]
            }
        else:
            if self.cfg.data_type == "latent":
                teacher_latent = batch["data1"].cuda()
                student_latent = batch["data2"].cuda()
                if "ae_feature1" in batch and self.cfg.model_type in ["wan_i2v", "qwen_image"]:
                    teacher_ae_feature = batch["ae_feature1"].cuda()
                    student_ae_feature = batch["ae_feature2"].cuda()
                    if self.cfg.model_type == "wan_i2v":
                        assert (
                            student_ae_feature.shape[2]
                            == teacher_ae_feature.shape[2] + self.cfg.teacher_latent_temporal_left_pad
                        )
                        assert self.cfg.teacher_wan_i2v.i2v_concat == self.cfg.student_wan_i2v.i2v_concat
                        if self.cfg.teacher_wan_i2v.i2v_concat:
                            bs = teacher_ae_feature.shape[0]
                            teacher_img_masks = self.image_encoder.get_image_mask(teacher_ae_feature, self.teacher_ae)
                            student_img_masks = self.image_encoder.get_image_mask(student_ae_feature, self.student_ae)
                            teacher_latent = torch.cat((teacher_latent, teacher_img_masks, teacher_ae_feature), dim=1)
                            student_latent = torch.cat((student_latent, student_img_masks, student_ae_feature), dim=1)
                    elif self.cfg.model_type == "qwen_image":
                        teacher_latent = torch.cat((teacher_latent, teacher_ae_feature), dim=1)
                        student_latent = torch.cat((student_latent, student_ae_feature), dim=1)
                    else:
                        raise NotImplementedError(f"Model type {self.cfg.model_type} not supported")

                if self.cfg.teacher_latent_temporal_left_pad > 0:
                    teacher_latent = torch.nn.functional.pad(
                        teacher_latent, (0, 0, 0, 0, self.cfg.teacher_latent_temporal_left_pad, 0), mode="replicate"
                    )
                if self.cfg.teacher_latent_temporal_right_trim > 0:
                    teacher_latent = teacher_latent[:, :, : -self.cfg.teacher_latent_temporal_right_trim]

                loss_dict = self.forward_with_latents(teacher_latent, student_latent, model)
            else:
                images = batch["image"].cuda()
                loss_dict = self.forward_with_images(images, model)

        if self.cfg.align_type == "average_pool":
            loss = loss_dict["avgpool_loss"]
        elif self.cfg.align_type == "upsample":
            loss = loss_dict[f"upsample_loss_{self.cfg.upsample_mode}"]
        elif self.cfg.align_type == "downsample":
            loss = loss_dict["downsample_loss"]
        elif self.cfg.align_type == "mixed":
            loss = (
                loss_dict["avgpool_loss"]
                + loss_dict[f"upsample_loss_{self.cfg.upsample_mode}"]
                + loss_dict["downsample_loss"]
            )
        else:
            raise NotImplementedError(f"alignment is not supported for {self.cfg.align_type}")

        if self.cfg.dataset == "latent_video":
            modes = ["nearest", "trilinear"]
        else:
            modes = ["nearest", "bilinear", "bicubic"]

        info = {
            "detailed_loss_dict": {
                "full_loss": loss.item(),
                "avgpool_loss": loss_dict["avgpool_loss"].item(),
                "downsample_loss": loss_dict["downsample_loss"].item(),
                **{f"upsample_loss_{mode}": loss_dict[f"upsample_loss_{mode}"].item() for mode in modes},
            },
        }

        return {0: loss}, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        if is_master():
            loss_dict_float = {key: value.item() for key, value in loss_dict.items()}
            if self.cfg.data_type == "latent":
                data_info = f"data1 {batch['data1'].sum()}, data2 {batch['data2'].sum()}"
            elif self.cfg.data_type == "image":
                data_info = f"images {batch['image'].sum()}"
            else:
                raise ValueError(f"data_type {self.cfg.data_type} is not supported")
            print(
                f"global step {self.global_step}, {data_info}, loss {loss_dict_float}, "
                f"grad_norm {info['grad_norm_0']}",
                flush=True,
            )

    @torch.no_grad()
    def evaluate(self, step: int, model: nn.Module, f_log=sys.stdout) -> dict[str, Any]:
        model.eval()
        data_loader = self.eval_data_providers[0].data_loader

        eval_loss_dict: dict[str, AverageMeter] = {}

        with tqdm(
            total=min(len(data_loader), self.cfg.eval_max_batch_num),
            desc="Valid Step #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            for idx, batch in enumerate(data_loader):
                _, eval_loss_info = self.model_forward(batch, model, training=False)
                batch_size = self.get_batch_size(batch)
                eval_loss_info = eval_loss_info["detailed_loss_dict"]
                for key in eval_loss_info:
                    if key not in eval_loss_dict:
                        eval_loss_dict[key] = AverageMeter()
                    eval_loss_dict[key].update(eval_loss_info[key], batch_size)
                t.set_postfix({key: eval_loss_dict[key].avg for key in eval_loss_dict})
                t.update()
                if idx >= self.cfg.eval_max_batch_num - 1:
                    break

        return {key: eval_loss_dict[key].avg for key in eval_loss_dict}

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        if self.cfg.data_type == "latent":
            return batch["data1"].shape[0]
        if self.cfg.data_type == "image":
            return batch["image"].shape[0]
        raise ValueError(f"data_type {self.cfg.data_type} is not supported")

    def get_current_step_train_loss_dict(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
        return info["detailed_loss_dict"]

    def get_train_data_provider_states(self, place_holder: bool = False) -> dict[str, Any]:
        if self.cfg.dataset in ["image_mixture", "latent_image", "latent_image_edit", "latent_video"]:
            return {
                "sampler_state_dict": self.train_data_provider.sampler.state_dict(
                    self.global_step,
                    place_holder=place_holder,
                )
            }
        elif self.cfg.dataset == "imagenet":
            return {}
        else:
            raise NotImplementedError(f"dataset {self.cfg.dataset} is not supported")

    def set_train_data_provider_states(self, train_data_provider_states: dict[str, Any]):
        if self.cfg.dataset in ["image_mixture", "latent_image", "latent_image_edit", "latent_video"]:
            self.train_data_provider.sampler.load_state_dict(train_data_provider_states["sampler_state_dict"])
        elif self.cfg.dataset == "imagenet":
            return
        else:
            raise NotImplementedError(f"dataset {self.cfg.dataset} is not supported")


def main():
    cfg: DCAdaptTrainerConfig = get_config(DCAdaptTrainerConfig)
    trainer = DCAdaptTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
