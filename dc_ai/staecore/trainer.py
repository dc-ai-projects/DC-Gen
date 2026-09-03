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

import copy
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from ..aecore.data_provider.base import AECoreDataProvider
from ..aecore.data_provider.collection import possible_eval_data_providers as possible_image_eval_data_providers
from ..aecore.data_provider.mixture import AECoreMixtureDataProvider, AECoreMixtureDataProviderConfig
from ..aecore.models.base import BaseAE
from ..aecore.trainer import AECoreTrainer, AECoreTrainerConfig
from ..apps.metrics.fid.fid import FIDStats
from ..apps.metrics.fvd.fvd import FVDStats, FVDStatsConfig
from ..apps.metrics.psnr.psnr import PSNRStats
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_barrier, is_master, sync_tensor
from ..apps.utils.metric import AverageMeter
from ..apps.utils.video import write_video
from ..stae_model_zoo import REGISTERED_DCAEV_MODEL
from .autoencoder import Autoencoder, AutoencoderConfig
from .data_provider.base import STAECoreDataProvider
from .data_provider.collection import possible_video_eval_data_providers
from .data_provider.mixture import STAECoreMixtureDataProvider, STAECoreMixtureDataProviderConfig
from .models.base import BaseSTAECoreModel
from .models.dc_ae_v import DCAEV, DCAEVConfig
from .models.wan_21_vae import Wan21VAE, Wan21VAEConfig
from .models.wan_22_vae import Wan22VAE, Wan22VAEConfig

__all__ = ["STAECoreTrainerConfig", "STAECoreTrainer"]


@dataclass
class STAECoreTrainerConfig(AECoreTrainerConfig):
    # evaluation
    output_fps: float = 8

    # eval data providers
    base_sample_size: tuple[int] = (256, 256, 1)  # base sample size for batch_size computation, H, W, T
    base_batch_size: int = 32

    # model
    wan_21_vae: Wan21VAEConfig = field(default_factory=Wan21VAEConfig)
    wan_22_vae: Wan22VAEConfig = field(default_factory=Wan22VAEConfig)
    dc_ae_v: DCAEVConfig = field(default_factory=DCAEVConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    latent_noise_std: Optional[float] = None

    # metrics
    compute_fid_with_jpeg: bool = False
    compute_metrics_for_last_n_frames: Optional[int] = None
    compute_fvd: bool = True
    fvd: FVDStatsConfig = field(default_factory=FVDStatsConfig)


class STAECoreTrainer(AECoreTrainer):
    def __init__(self, cfg: STAECoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: STAECoreTrainerConfig
        self.model: BaseSTAECoreModel

    def build_eval_data_providers(self) -> list[AECoreDataProvider]:
        base_sample_volume = np.prod(self.cfg.base_sample_size).item()

        eval_data_providers: list[AECoreDataProvider] = []
        for eval_data_provider_name_and_size in self.cfg.eval_data_providers:
            eval_data_provider_name_and_size_splitted = eval_data_provider_name_and_size.split("_")
            eval_data_provider_name = eval_data_provider_name_and_size_splitted[0]
            if eval_data_provider_name in possible_image_eval_data_providers:
                resolution = int(eval_data_provider_name_and_size_splitted[1])
                batch_size = max(self.cfg.base_batch_size * base_sample_volume // resolution**2, 1)
                data_provider_cfg = possible_image_eval_data_providers[eval_data_provider_name][0](
                    resolution=resolution, batch_size=batch_size
                )
                data_provider = possible_image_eval_data_providers[eval_data_provider_name][1](data_provider_cfg)
                if self.cfg.compute_fid and data_provider_cfg.fid_ref_path is not None:
                    assert os.path.exists(
                        data_provider_cfg.fid_ref_path
                    ), f"{data_provider_cfg.fid_ref_path} not existed"
            elif eval_data_provider_name in possible_video_eval_data_providers:
                h, w, t = map(int, eval_data_provider_name_and_size_splitted[1:4])
                batch_size = max(self.cfg.base_batch_size * base_sample_volume // h // w // t, 1)
                kwargs = {"h": h, "w": w, "t": t, "batch_size": batch_size}
                if len(eval_data_provider_name_and_size_splitted) >= 5:
                    kwargs["fps"] = eval_data_provider_name_and_size_splitted[4]
                data_provider_cfg = possible_video_eval_data_providers[eval_data_provider_name][0](**kwargs)
                data_provider = possible_video_eval_data_providers[eval_data_provider_name][1](data_provider_cfg)
                if self.cfg.compute_fvd and data_provider_cfg.fvd_ref_path is not None:
                    assert os.path.exists(
                        data_provider_cfg.fvd_ref_path
                    ), f"{data_provider_cfg.fvd_ref_path} not existed"
            else:
                raise ValueError(f"eval data provider {eval_data_provider_name} is not supported")
            eval_data_providers.append(data_provider)
        return eval_data_providers

    def get_possible_models(self) -> dict[str, type[BaseAE]]:
        possible_models = super().get_possible_models()
        possible_models["wan_21_vae"] = partial(Wan21VAE, self.cfg.wan_21_vae)
        possible_models["wan_22_vae"] = partial(Wan22VAE, self.cfg.wan_22_vae)
        possible_models["dc_ae_v"] = partial(DCAEV, self.cfg.dc_ae_v)
        for model_name, (model_cfg_func, pretrained_path) in REGISTERED_DCAEV_MODEL.items():
            possible_models[model_name] = (
                lambda model_cfg_func=model_cfg_func, model_name=model_name, pretrained_path=pretrained_path: (
                    DCAEV(model_cfg_func(model_name, pretrained_path))
                )
            )  # To avoid "cell variable defined in loop": https://pylint.readthedocs.io/en/latest/user_guide/messages/warning/cell-var-from-loop.html
        possible_models["autoencoder"] = partial(Autoencoder, self.cfg.autoencoder)
        return possible_models

    @torch.no_grad()
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
            from torchmetrics.image import StructuralSimilarityIndexMeasure

            ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0)).to(device)
        if self.cfg.compute_lpips:
            from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

            lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
        compute_fvd = self.cfg.compute_fvd and data_provider.cfg.fvd_ref_path is not None
        if compute_fvd:
            assert os.path.exists(data_provider.cfg.fvd_ref_path), f"{data_provider.cfg.fvd_ref_path} not existed"
            fvd_stats = FVDStats(self.cfg.fvd)

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
            num_saved_samples = 0
            for _, batch in enumerate(data_provider.data_loader):
                if "image" in batch:
                    samples: torch.Tensor = batch["image"]
                    sample_kind = "image"
                elif "video" in batch:
                    samples: torch.Tensor = batch["video"]
                    sample_kind = "video"
                else:
                    raise ValueError(f"batch does not contain 'image' or 'video' key")
                # preprocessing
                samples = samples.cuda().to(self.amp_dtype)
                # forward

                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                    kwargs = {}
                    if latent_channels is not None:
                        kwargs["latent_channels"] = latent_channels
                    if sample_kind == "image":
                        output, info = model("reconstruct_image", samples)
                    elif sample_kind == "video":
                        if self.cfg.latent_noise_std is None:
                            output, info = model("reconstruct_video", samples, **kwargs)
                        else:
                            latent = model("encode", samples)
                            output = model("decode", latent + self.cfg.latent_noise_std * torch.randn_like(latent))
                            info = {"latent": latent}
                    else:
                        raise ValueError(f"sample kind {sample_kind} is not supported")

                input_samples = samples * 0.5 + 0.5
                output_samples = output * 0.5 + 0.5
                # save samples
                if (
                    num_saved_samples < self.cfg.num_save_samples
                    and (is_master() or self.cfg.save_samples_at_all_ranks)
                ) or self.cfg.save_all_samples:
                    for j in range(input_samples.shape[0]):
                        if sample_kind == "image":
                            save_image(
                                torch.cat([input_samples[j : j + 1], output_samples[j : j + 1]], dim=3),
                                os.path.join(eval_dir, f"{self.rank}_{num_saved_samples}.jpg"),
                            )
                        elif sample_kind == "video":
                            write_video(
                                os.path.join(eval_dir, f"{self.rank}_{num_saved_samples}.mp4"),
                                torch.cat([input_samples[j], output_samples[j]], dim=3),
                                fps=self.cfg.output_fps,
                            )
                        else:
                            raise ValueError(f"sample kind {sample_kind} is not supported")
                        num_saved_samples += 1
                        if num_saved_samples >= self.cfg.num_save_samples and not self.cfg.save_all_samples:
                            break
                # update metrics
                if "detailed_loss_dict" in info:
                    for loss_key, loss_value in info["detailed_loss_dict"].items():
                        if loss_key not in eval_loss_dict:
                            eval_loss_dict[loss_key] = AverageMeter()
                        eval_loss_dict[loss_key].update(loss_value, samples.shape[0])
                if sample_kind == "image":
                    input_images_uint8 = (255 * input_samples + 0.5).clamp(0, 255).to(torch.uint8)
                    output_images_uint8 = (255 * output_samples + 0.5).clamp(0, 255).to(torch.uint8)
                elif sample_kind == "video":
                    _, C, _, H, W = samples.shape
                    input_videos_uint8 = (255 * input_samples + 0.5).clamp(0, 255).to(torch.uint8)  # (B, 3, T, H, W)
                    output_videos_uint8 = (255 * output_samples + 0.5).clamp(0, 255).to(torch.uint8)  # (B, 3, T, H, W)
                    if self.cfg.compute_metrics_for_last_n_frames is not None:
                        input_videos_uint8 = input_videos_uint8[:, :, -self.cfg.compute_metrics_for_last_n_frames :]
                        output_videos_uint8 = output_videos_uint8[:, :, -self.cfg.compute_metrics_for_last_n_frames :]
                    input_images_uint8 = input_videos_uint8.transpose(1, 2).reshape(
                        -1, C, H, W
                    )  # (B, 3, T, H, W) -> (B * T, 3, H, W)
                    output_images_uint8 = output_videos_uint8.transpose(1, 2).reshape(-1, C, H, W)
                else:
                    raise ValueError(f"sample kind {sample_kind} is not supported")

                if compute_fid:
                    assert sample_kind == "image"

                    if self.cfg.compute_fid_with_jpeg:
                        output_images_jpeg = []
                        for image in output_images_uint8:
                            with BytesIO() as buff:
                                Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(buff, format="JPEG")
                                buff.seek(0)
                                out = buff.read()
                                output_images_jpeg.append(
                                    torch.tensor(np.array(Image.open(BytesIO(out)))).permute(2, 0, 1).cuda()
                                )
                        output_images_uint8 = torch.stack(output_images_jpeg)

                    fid_stats.add_data(output_images_uint8)
                if self.cfg.compute_psnr:
                    psnr.add_data(input_images_uint8, output_images_uint8)
                if self.cfg.compute_ssim:
                    ssim.update(input_images_uint8, output_images_uint8)
                if self.cfg.compute_lpips:
                    lpips.update(input_images_uint8 / 255, output_images_uint8 / 255)
                if compute_fvd:
                    fvd_stats.add_data(output_videos_uint8)

                if self.cfg.get_latent_stats:
                    latent = info["latent"]
                    assert not torch.any(torch.isnan(latent))
                    if self.cfg.get_per_channel_latent_stats:
                        latent_total_sum += latent.float().sum(dim=(0, 2, 3, 4)).cpu().numpy()
                        latent_total_sum_squared += latent.float().square().sum(dim=(0, 2, 3, 4)).cpu().numpy()
                        latent_total_cnt += latent[:, 0].numel()
                    else:
                        latent_total_sum += latent.float().sum().item()
                        latent_total_sum_squared += latent.float().square().sum().item()
                        latent_total_cnt += latent.numel()
                # tqdm
                postfix_dict = {
                    "bs": samples.shape[0],
                    "res": samples.shape[3],
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
            eval_info_dict["latent_rms_reverse"] = 1 / rms
            eval_info_dict["latent_std"] = std
        if compute_fid:
            eval_info_dict["fid"] = fid_stats.compute_fid(data_provider.cfg.fid_ref_path)
        if self.cfg.compute_psnr:
            eval_info_dict.update(psnr.compute())
        if self.cfg.compute_ssim:
            eval_info_dict["ssim"] = ssim.compute().item()
        if self.cfg.compute_lpips:
            eval_info_dict["lpips"] = lpips.compute().item()
        if compute_fvd:
            eval_info_dict["fvd"] = fvd_stats.compute_fvd(data_provider.cfg.fvd_ref_path)
        return eval_info_dict


def main():
    cfg: STAECoreTrainerConfig = get_config(STAECoreTrainerConfig)
    trainer = STAECoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
