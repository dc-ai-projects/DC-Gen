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
from dataclasses import dataclass, field
from typing import Any, Optional

import ipdb
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image

os.environ["TOKENIZERS_PARALLELISM"] = (
    "true"  # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
)
import sys
import time
from functools import partial

import pandas
from omegaconf import MISSING
from PIL import Image, ImageFile
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torchvision.utils import save_image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

from ..ae_model_zoo import DCAE_HF, REGISTERED_DCAE_MODEL, REGISTERED_SD_VAE_MODEL
from ..apps.metrics.clip_score import CLIPScoreStats
from ..apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from ..apps.metrics.image_reward import ImageRewardStats, ImageRewardStatsConfig
from ..apps.metrics.knn.self_knn import SelfKnnStats, SelfKnnStatsConfig
from ..apps.metrics.psnr.psnr import PSNRStats, PSNRStatsConfig
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_barrier, is_dist_initialized, is_master, sync_tensor
from ..apps.utils.metric import AverageMeter
from ..models.utils.network import get_params_num
from .data_provider.base import AECoreDataProvider
from .data_provider.collection import possible_eval_data_providers
from .data_provider.mixture import AECoreMixtureDataProvider, AECoreMixtureDataProviderConfig
from .models.base import BaseAE
from .models.dc_ae import DCAE
from .models.dc_ae_diffusers import DCAEDiffusers, DCAEDiffusersConfig
from .models.hf_autoencoder_kl import HFAutoencoderKL, HFAutoencoderKLConfig
from .models.sd_vae import SDVAE, SDVAEConfig

__all__ = ["AECoreTrainerConfig", "AECoreTrainer"]


@dataclass
class AECoreTrainerConfig(BaseTrainerConfig):
    # eval
    get_latent_stats: bool = True
    get_per_channel_latent_stats: bool = False

    # eval data providers
    eval_data_providers: tuple[str, ...] = ()
    base_sample_size: int = 256
    base_batch_size: int = 32
    adaptive_latent_channels: Optional[tuple[int]] = None
    eval_adaptive_latent_channels: Optional[tuple[int]] = "${.adaptive_latent_channels}"

    # model
    model: str = MISSING
    hf_autoencoder_kl: HFAutoencoderKLConfig = field(default_factory=HFAutoencoderKLConfig)
    sd_vae: SDVAEConfig = field(default_factory=SDVAEConfig)
    dc_ae_diffusers: DCAEDiffusersConfig = field(default_factory=DCAEDiffusersConfig)

    # metrics
    compute_fid: bool = True
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    compute_psnr: bool = True
    psnr: PSNRStatsConfig = field(default_factory=PSNRStatsConfig)
    compute_ssim: bool = True
    compute_lpips: bool = True
    compute_clip_iqa: bool = True
    compute_clip_score: bool = True
    compute_image_reward: bool = True
    image_reward: ImageRewardStatsConfig = field(default_factory=ImageRewardStatsConfig)
    compute_self_knn: bool = True
    self_knn: SelfKnnStatsConfig = field(default_factory=SelfKnnStatsConfig)

    # process data
    convert_to_amp_dtype_before_forward: bool = True
    num_sample_latent_channels: int = 1
    always_sample_min_max_channel: bool = False
    always_sample_max_channel: bool = False

    # train data providers
    mixture: AECoreMixtureDataProviderConfig = field(
        default_factory=lambda: AECoreMixtureDataProviderConfig(save_checkpoint_steps="${..save_checkpoint_steps}")
    )


class AECoreTrainer(BaseTrainer):
    def __init__(self, cfg: AECoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: AECoreTrainerConfig
        self.model: BaseAE
        if cfg.mode == "train":
            self.train_data_provider: AECoreMixtureDataProvider

    def build_eval_data_providers(self) -> list[AECoreDataProvider]:
        eval_data_providers: list[AECoreDataProvider] = []
        for eval_data_provider_name_and_resolution in self.cfg.eval_data_providers:
            eval_data_provider_name, resolution = eval_data_provider_name_and_resolution.split("_")
            resolution = int(resolution)
            batch_size = max(self.cfg.base_batch_size * self.cfg.base_sample_size**2 // resolution**2, 1)
            if eval_data_provider_name in possible_eval_data_providers:
                data_provider_cfg = possible_eval_data_providers[eval_data_provider_name][0](
                    resolution=resolution, batch_size=batch_size
                )
                data_provider = possible_eval_data_providers[eval_data_provider_name][1](data_provider_cfg)
            else:
                raise ValueError(f"eval data provider {eval_data_provider_name} is not supported")
            eval_data_providers.append(data_provider)
        return eval_data_providers

    def get_possible_models(self) -> dict[str, type[BaseAE]]:
        possible_models = {}
        for model_name, (model_cfg_func, pretrained_path, organization) in REGISTERED_DCAE_MODEL.items():
            if pretrained_path is None:
                possible_models[model_name] = partial(DCAE_HF.from_pretrained, f"{organization}/{model_name}")
            else:
                cfg = model_cfg_func(model_name, pretrained_path)
                possible_models[model_name] = partial(DCAE, cfg)
        for model_name, (model_cfg_func, pretrained_path) in REGISTERED_SD_VAE_MODEL.items():
            possible_models[model_name] = partial(model_cfg_func, name=model_name, pretrained_path=pretrained_path)
        possible_models["hf_autoencoder_kl"] = partial(HFAutoencoderKL, self.cfg.hf_autoencoder_kl)
        possible_models["sd_vae"] = partial(SDVAE, self.cfg.sd_vae)
        possible_models["dc_ae_diffusers"] = partial(DCAEDiffusers, self.cfg.dc_ae_diffusers)
        return possible_models

    def build_model(self) -> BaseAE:
        possible_models = self.get_possible_models()
        if self.cfg.model in possible_models:
            model = possible_models[self.cfg.model]()
        else:
            raise ValueError(f"model {self.cfg.model} is not supported among {possible_models.keys()}")
        if is_master():
            print(f"training params: {get_params_num(model):.2f} M")
            print(f"all params: {get_params_num(model, train_only=False):.2f} M")
        return model

    def evaluate_single(
        self,
        data_provider: AECoreDataProvider,
        step: int,
        model: BaseAE,
        latent_channels: Optional[int] = None,
        f_log=sys.stdout,
        additional_dir_name: str = "",
    ) -> dict[str, Any]:
        model.eval()
        eval_loss_dict: dict[str, AverageMeter] = dict()
        device = torch.device("cuda")

        # metrics
        compute_fid = self.cfg.compute_fid and data_provider.cfg.fid_ref_path is not None
        if compute_fid:
            assert os.path.exists(data_provider.cfg.fid_ref_path)
            fid_stats = FIDStats(self.cfg.fid)
        if self.cfg.compute_psnr:
            psnr = PSNRStats(self.cfg.psnr)
        if self.cfg.compute_ssim:
            ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0)).to(device)
        if self.cfg.compute_lpips:
            lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        if self.cfg.compute_clip_iqa:
            clip_iqa = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        if self.cfg.compute_clip_score:
            clip_score = CLIPScoreStats()
        if self.cfg.compute_image_reward:
            image_reward = ImageRewardStats(self.cfg.image_reward)
        if self.cfg.compute_self_knn:
            self_knn_stats = SelfKnnStats(self.cfg.self_knn)

        if self.cfg.get_latent_stats:
            latent_total_sum = 0
            latent_total_sum_squared = 0
            latent_total_cnt = 0

        if self.cfg.eval_dir_name is not None:
            eval_dir = os.path.join(self.cfg.run_dir, self.cfg.eval_dir_name, additional_dir_name)
        else:
            eval_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(eval_dir, exist_ok=True)
        dist_barrier()

        with tqdm(
            total=len(data_provider.data_loader),
            desc="eval Steps #{}".format(step),
            disable=not is_master(),
            file=f_log,
            mininterval=10.0,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ) as t:
            num_saved_images = 0
            for _, (images, labels) in enumerate(data_provider.data_loader):
                # preprocessing
                images = images.cuda().to(self.amp_dtype)
                # forward
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    output, _, info = model.reconstruct_image(images, latent_channels=latent_channels)

                input_images = images * 0.5 + 0.5
                output_images = output * 0.5 + 0.5
                # save samples
                if (
                    num_saved_images < self.cfg.num_save_samples and (is_master() or self.cfg.save_samples_at_all_ranks)
                ) or self.cfg.save_all_samples:
                    for j in range(input_images.shape[0]):
                        save_image(
                            torch.cat([input_images[j : j + 1], output_images[j : j + 1]], dim=3),
                            os.path.join(eval_dir, f"{self.rank}_{num_saved_images}.jpg"),
                        )
                        num_saved_images += 1
                        if num_saved_images >= self.cfg.num_save_samples and not self.cfg.save_all_samples:
                            break
                # update metrics
                if "detailed_loss_dict" in info:
                    for loss_key, loss_value in info["detailed_loss_dict"].items():
                        if loss_key not in eval_loss_dict:
                            eval_loss_dict[loss_key] = AverageMeter(is_distributed=is_dist_initialized())
                        eval_loss_dict[loss_key].update(loss_value.item(), images.shape[0])
                input_images_uint8 = (255 * input_images + 0.5).clamp(0, 255).to(torch.uint8)
                output_images_uint8 = (255 * output_images + 0.5).clamp(0, 255).to(torch.uint8)
                if compute_fid:
                    fid_stats.add_data(output_images_uint8)
                if self.cfg.compute_psnr:
                    psnr.add_data(input_images_uint8, output_images_uint8)
                if self.cfg.compute_ssim:
                    ssim.update(input_images_uint8, output_images_uint8)
                if self.cfg.compute_lpips:
                    lpips.update(input_images_uint8 / 255, output_images_uint8 / 255)
                if self.cfg.compute_clip_iqa:
                    clip_iqa.update(output_images_uint8.float())
                if self.cfg.compute_clip_score and "vlm_caption" in labels:
                    clip_score.update(output_images_uint8, labels["vlm_caption"])
                if self.cfg.compute_image_reward and "vlm_caption" in labels:
                    image_reward.add_data(output_images_uint8, labels["vlm_caption"])
                if self.cfg.compute_self_knn and data_provider.cfg.name == "ImageNetEval":
                    latent: torch.Tensor = info["latent"]
                    class_label: torch.Tensor = labels["class_label"].cuda()
                    self_knn_stats.add_data(latent.mean(dim=[2, 3]), class_label)
                if self.cfg.get_latent_stats:
                    latent: torch.Tensor = info["latent"]
                    assert not torch.any(torch.isnan(latent))
                    if self.cfg.get_per_channel_latent_stats:
                        latent_total_sum = latent_total_sum + latent.float().sum(dim=(0, 2, 3)).cpu().numpy()
                        latent_total_sum_squared = (
                            latent_total_sum_squared + latent.float().square().sum(dim=(0, 2, 3)).cpu().numpy()
                        )
                        latent_total_cnt += latent[:, 0].numel()
                    else:
                        latent_total_sum += latent.float().sum().item()
                        latent_total_sum_squared += latent.float().square().sum().item()
                        latent_total_cnt += latent.numel()
                # tqdm
                postfix_dict = {
                    "bs": images.shape[0],
                    "res": images.shape[2],
                }
                for key in eval_loss_dict:
                    postfix_dict[key] = eval_loss_dict[key].avg
                t.set_postfix(postfix_dict, refresh=False)
                t.update()
        eval_info_dict = {key: value.avg for key, value in eval_loss_dict.items()}
        torch.cuda.empty_cache()
        if self.cfg.get_latent_stats:
            latent_total_sum = sync_tensor(torch.tensor(latent_total_sum).cuda(), reduce="sum").cpu().numpy()
            latent_total_sum_squared = (
                sync_tensor(torch.tensor(latent_total_sum_squared).cuda(), reduce="sum").cpu().numpy()
            )
            latent_total_cnt = sync_tensor(torch.tensor(latent_total_cnt).cuda(), reduce="sum").cpu().numpy()
            mean = latent_total_sum / latent_total_cnt
            rms = np.sqrt(latent_total_sum_squared / latent_total_cnt)
            variance = (latent_total_sum_squared - mean * latent_total_sum) / (latent_total_cnt - 1)
            std = np.sqrt(variance)
            eval_info_dict["latent_mean"] = mean
            eval_info_dict["latent_rms"] = rms
            eval_info_dict["latent_std"] = std
        if compute_fid:
            eval_info_dict["fid"] = fid_stats.compute_fid(data_provider.cfg.fid_ref_path)
        if self.cfg.compute_psnr:
            eval_info_dict["psnr"] = psnr.compute()
        if self.cfg.compute_ssim:
            eval_info_dict["ssim"] = ssim.compute().item()
        if self.cfg.compute_lpips:
            eval_info_dict["lpips"] = lpips.compute().item()
        if self.cfg.compute_clip_iqa:
            eval_info_dict["clip_iqa"] = clip_iqa.compute().mean().item()
        if self.cfg.compute_clip_score and clip_score.n_samples > 0:
            eval_info_dict["clip_score"] = clip_score.compute()
        if self.cfg.compute_image_reward and image_reward.cnt > 0:
            eval_info_dict["image_reward"] = image_reward.compute()
        if self.cfg.compute_self_knn and len(self_knn_stats.labels_rank) > 0:
            eval_info_dict.update(self_knn_stats.compute())
        return eval_info_dict

    @torch.no_grad()
    def evaluate(self, step: int, model: BaseAE, f_log=sys.stdout) -> dict[str, Any]:
        results_path = os.path.join(self.cfg.run_dir, "eval_results.csv")
        if os.path.exists(results_path):
            results = pandas.read_csv(results_path, index_col=0)
        else:
            results = pandas.DataFrame()
        eval_info_dict = {}
        possible_latent_channels = (
            self.cfg.eval_adaptive_latent_channels if self.cfg.eval_adaptive_latent_channels is not None else [None]
        )
        for latent_channels in possible_latent_channels:
            for data_provider_name, data_provider in zip(self.cfg.eval_data_providers, self.eval_data_providers):
                eval_dir_name = f"{self.cfg.eval_dir_name}" if self.cfg.eval_dir_name is not None else f"step_{step}"
                setting = ""
                if latent_channels is not None:
                    setting += f"latent_channels_{latent_channels}_"
                setting += f"{data_provider_name}"
                index = f"{eval_dir_name}_{setting}"
                if index in results.index:
                    eval_info_dict[setting] = results.loc[[index]].to_dict(orient="index")[index]
                else:
                    eval_info_dict[setting] = self.evaluate_single(
                        data_provider=data_provider,
                        step=step,
                        model=model,
                        latent_channels=latent_channels,
                        f_log=f_log,
                        additional_dir_name=setting,
                    )
                    if is_master():
                        results = pandas.concat(
                            [results, pandas.DataFrame.from_dict({index: eval_info_dict[setting]}, orient="index")]
                        ).sort_index()
                        results.to_csv(results_path)
        return eval_info_dict

    def build_train_data_provider(self) -> AECoreMixtureDataProvider:
        return AECoreMixtureDataProvider(self.cfg.mixture)

    def setup_model_for_training(self):
        if self.cfg.model in ["dc_ae_train", "sd_vae"]:
            self.model.convert_sync_batchnorm()  # https://github.com/huggingface/pytorch-image-models/issues/1254
        else:
            raise ValueError(f"model {self.cfg.model} is not supported for training")
        super().setup_model_for_training()

    def get_trainable_module_list(self, model: BaseAE) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def load_fsdp_model_to_eval(self, model_to_eval: nn.Module, eval_checkpoint_path: str):
        raise NotImplementedError

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().prepare_batch(batch)
        if self.cfg.adaptive_latent_channels is not None:
            if self.cfg.num_sample_latent_channels == 1:
                latent_channels = self.cfg.adaptive_latent_channels[
                    torch.randint(
                        0,
                        len(self.cfg.adaptive_latent_channels),
                        (1,),
                        generator=self.train_async_generator_gpu,
                        device=self.device,
                    ).item()
                ]
            elif self.cfg.always_sample_min_max_channel:
                indices = torch.randint(
                    0,
                    len(self.cfg.adaptive_latent_channels),
                    (self.cfg.num_sample_latent_channels - 2,),
                    generator=self.train_async_generator_gpu,
                    device=self.device,
                )
                latent_channels = [self.cfg.adaptive_latent_channels[0], self.cfg.adaptive_latent_channels[-1]] + [
                    self.cfg.adaptive_latent_channels[index.item()] for index in indices
                ]
            elif self.cfg.always_sample_max_channel:
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
        else:
            latent_channels = None
        batch["latent_channels"] = latent_channels
        return batch

    def model_forward(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
            kwargs = {}
            if batch["latent_channels"] is not None:
                kwargs["latent_channels"] = batch["latent_channels"]
            output, loss_dict, info = self.model(
                batch["images"],
                self.global_step,
                sync_generator=self.train_sync_generator_gpu,
                async_generator=self.train_async_generator_gpu,
                **kwargs,
            )
            info["output"] = output
            if "detailed_loss_dict" in info:
                for key in info["detailed_loss_dict"]:
                    info["detailed_loss_dict"][key] = info["detailed_loss_dict"][key].item()
        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        if is_master():
            loss_dict_float = {key: value.item() for key, value in loss_dict.items()}
            print(
                f"global step {self.global_step}, images {batch['images'].sum()}, loss {loss_dict_float}, grad_norm {info['grad_norm_0'] if 'grad_norm_0' in info else info['grad_norm_1']}",
                flush=True,
            )

    def save_samples(self, batch: dict[str, Any], info: dict[str, Any]):
        input_images = batch["images"] * 0.5 + 0.5
        save_image(
            input_images,
            os.path.join(
                self.train_save_samples_dir,
                f"{self.global_step:08d}{f'_{self.rank}' if self.rank != 0 else ''}_input_images.jpg",
            ),
            nrow=int(np.sqrt(input_images.shape[0])),
        )
        if isinstance(info["output"], torch.Tensor):
            output_images = info["output"] * 0.5 + 0.5
            save_image(
                output_images,
                os.path.join(
                    self.train_save_samples_dir,
                    f"{self.global_step:08d}{f'_{self.rank}' if self.rank != 0 else ''}_output_images.jpg",
                ),
                nrow=int(np.sqrt(output_images.shape[0])),
            )
        elif isinstance(info["output"], list) and all(isinstance(output_, torch.Tensor) for output_ in info["output"]):
            for i, output_ in enumerate(info["output"]):
                output_images = output_ * 0.5 + 0.5
                save_image(
                    output_images,
                    os.path.join(
                        self.train_save_samples_dir,
                        f"{self.global_step:08d}{f'_{self.rank}' if self.rank != 0 else ''}_output_images_{batch['latent_channels'][i]}.jpg",
                    ),
                    nrow=int(np.sqrt(output_images.shape[0])),
                )
        else:
            raise ValueError(f"output {type(info['output'])} is not supported")

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
        postfix_dict["res"] = log_dict["train/resolution"] = batch["images"].shape[2]
        postfix_dict["latent_channels"] = (
            batch["latent_channels"] if batch["latent_channels"] is not None else info["latent"].shape[1]
        )

        # latent stats
        if self.cfg.get_latent_stats:
            latent = info["latent"]
            if isinstance(latent, torch.Tensor):
                log_dict["train/latent_mean"] = latent.mean().item()
                log_dict["train/latent_mean_square"] = latent.square().mean().item()
            elif isinstance(latent, list) and all(isinstance(latent_, torch.Tensor) for latent_ in latent):
                log_dict["train/latent_mean"] = np.mean([latent_.mean().item() for latent_ in latent])
                log_dict["train/latent_mean_square"] = np.mean([latent_.square().mean().item() for latent_ in latent])
            else:
                raise ValueError(f"latent {type(latent)} is not supported")

    def run_eval(self):
        start_time = time.time()
        eval_info_dict = self.evaluate(0, self.model)
        if is_master():
            print(eval_info_dict)
            for setting in eval_info_dict:
                print(f"setting {setting}")
                keys, values = [], []
                for key, value in eval_info_dict[setting].items():
                    if not isinstance(value, float) or not np.isnan(value):
                        keys.append(key)
                        values.append(value)
                print(", ".join(keys))
                print(", ".join([f"{value:.4f}" for value in values]))
            print(f"evaluation time: {time.time() - start_time:.2f}s")


def main():
    cfg: AECoreTrainerConfig = get_config(AECoreTrainerConfig)
    trainer = AECoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
