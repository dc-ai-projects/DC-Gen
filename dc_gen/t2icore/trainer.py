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

from ..apps.metrics.clip_score import CLIPScoreStats
from ..apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.utils import AverageMeter
from ..apps.utils.dist import dist_barrier, get_dist_rank, is_dist_initialized, is_master, sync_tensor
from ..c2icore.autoencoder import Autoencoder, AutoencoderConfig
from ..models.utils.network import freeze_weights, get_dtype_from_str, get_params_num
from .data_provider.base_eval import T2ICoreEvalDataProvider
from .data_provider.latent_mixture import T2ICoreLatentMixtureDataProvider, T2ICoreLatentMixtureDataProviderConfig
from .data_provider.mjhq_text_prompt import MJHQTextPromptDataProvider, MJHQTextPromptDataProviderConfig
from .models.base import BaseT2IModel, BaseT2IModelConfig
from .text_encoder import SingleTextEncoderConfig, TextEncoder

__all__ = ["T2ICoreTrainerConfig", "T2ICoreTrainer"]


@dataclass
class T2ICoreTrainerConfig(BaseTrainerConfig):
    # env
    resolution: int = 512
    save_image_format: str = "png"
    cfg_scale: float = 4.5
    pag_scale: float = 1.0

    # text encoder
    text_encoders: dict[str, SingleTextEncoderConfig] = field(
        default_factory=lambda: dict(),
    )

    # eval data providers
    eval_data_providers: tuple[str, ...] = ()
    mjhq_text_prompt: MJHQTextPromptDataProviderConfig = field(
        default_factory=lambda: MJHQTextPromptDataProviderConfig(resolution="${..resolution}")
    )

    # train data providers
    train_data_scaling_factor: Any = None
    mixture: T2ICoreLatentMixtureDataProviderConfig = field(
        default_factory=lambda: T2ICoreLatentMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}"
        )
    )

    # autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    autoencoder_dtype: str = "fp32"

    # model
    model: str = MISSING
    model_dtype: Optional[str] = None

    # metrics
    compute_fid: bool = True
    compute_clip_score: bool = True
    compute_metrics_with_jpeg: bool = True

    # dc-ae 1.5
    adaptive_latent_channels: Optional[tuple[int]] = None
    num_sample_latent_channels: int = 1
    always_sample_max_channel: bool = False

    # steps
    save_samples_steps: int = 500
    save_checkpoint_steps: int = 500
    eval_steps: int = 5000


class T2ICoreTrainer(BaseTrainer):
    def __init__(self, cfg: T2ICoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: T2ICoreTrainerConfig

    def build_eval_data_providers(self) -> list[T2ICoreEvalDataProvider]:
        eval_data_providers = []
        for eval_data_provider_name in self.cfg.eval_data_providers:
            if eval_data_provider_name == "MJHQTextPrompt":
                eval_data_providers.append(MJHQTextPromptDataProvider(self.cfg.mjhq_text_prompt))
            else:
                raise ValueError(f"eval data provider {eval_data_provider_name} is not supported")
        return eval_data_providers

    def build_train_data_provider(self) -> T2ICoreLatentMixtureDataProvider:
        train_data_provider = T2ICoreLatentMixtureDataProvider(self.cfg.mixture)

        if self.cfg.train_data_scaling_factor is None:
            self.train_data_scaling_factor = None
        elif isinstance(self.cfg.train_data_scaling_factor, float):
            self.train_data_scaling_factor = self.cfg.train_data_scaling_factor
        elif isinstance(self.cfg.train_data_scaling_factor, str):
            self.train_data_scaling_factor = torch.tensor(
                np.load(self.cfg.train_data_scaling_factor), dtype=torch.float32, device=self.device
            )
        else:
            raise ValueError(f"train_data_scaling_factor {self.cfg.train_data_scaling_factor} is not supported")

        return train_data_provider

    def get_possible_models(self) -> dict[str, tuple[BaseT2IModelConfig, type[BaseT2IModel]]]:
        raise NotImplementedError

    def build_model(self) -> BaseT2IModel:
        possible_models = self.get_possible_models()

        if self.cfg.model in possible_models:
            model_cfg, model_class = possible_models[self.cfg.model]
            model_cfg.input_size = self.cfg.resolution // self.autoencoder.spatial_compression_ratio
            model = model_class(model_cfg)
        else:
            raise ValueError(f"model {self.cfg.model} is not supported among {possible_models.keys()}")

        if self.cfg.model_dtype is not None:
            model = model.to(dtype=get_dtype_from_str(self.cfg.model_dtype))

        if is_master():
            print(f"params: {get_params_num(model):.2f} M")

        return model

    def setup_model(self) -> None:
        self.text_encoder = TextEncoder(self.cfg.text_encoders).to(device=self.device)
        freeze_weights(self.text_encoder)

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
    def evaluate_single(
        self,
        step: int,
        network: BaseT2IModel,
        f_log=sys.stdout,
        cfg_scale: float = 4.5,
        pag_scale: float = 1.0,
        additional_dir_name: Optional[str] = None,
    ) -> dict[str, Any]:
        network.eval()
        eval_generator = torch.Generator(device=torch.device("cuda"))
        eval_generator.manual_seed(self.cfg.seed + self.rank)

        data_provider = self.eval_data_providers[0]
        data_loader = data_provider.data_loader

        # metrics
        compute_fid = self.cfg.compute_fid and data_provider.cfg.fid_ref_path is not None
        if compute_fid:
            assert os.path.exists(os.path.expanduser(data_provider.cfg.fid_ref_path))
            fid_stats = FIDStats(FIDStatsConfig(ref_path=data_provider.cfg.fid_ref_path))
        if self.cfg.compute_clip_score:
            clip_score_stats = CLIPScoreStats()

        if self.cfg.eval_dir_name is not None:
            eval_dir = os.path.join(self.cfg.run_dir, self.cfg.eval_dir_name, additional_dir_name)
        else:
            eval_dir = os.path.join(self.cfg.run_dir, f"{step}", additional_dir_name)
        if is_master():
            os.makedirs(eval_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        def update_metrics(image_samples_uint8, texts):
            if compute_fid:
                fid_stats.add_data(image_samples_uint8)
            if self.cfg.compute_clip_score:
                clip_score_stats.update(image_samples_uint8, texts)

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
                prompts = samples["prompt"]
                text_embed_info = self.text_encoder.get_text_embed_info(prompts, self.device)

                with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    latent_samples = network.generate(
                        text_embed_info=text_embed_info,
                        cfg_scale=cfg_scale,
                        pag_scale=pag_scale,
                        generator=eval_generator,
                    )

                image_samples = self.autoencoder.decode(
                    latent_samples.to(get_dtype_from_str(self.cfg.autoencoder_dtype))
                )

                # assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
                image_samples_numpy = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()

                if (
                    num_saved_samples < self.cfg.num_save_samples
                    and (is_master() or self.cfg.save_samples_at_all_ranks)
                ) or self.cfg.save_all_samples:
                    names = samples["name"]
                    image_samples_PIL = [Image.fromarray(image) for image in image_samples_numpy]
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

                update_metrics(image_samples_uint8, prompts)

                if compute_fid:
                    fid_res = fid_stats.compute_fid()
                    if is_master():
                        print("FID:", fid_res)
                # clip-score
                if self.cfg.compute_clip_score:
                    clip_res = clip_score_stats.compute()
                    if is_master():
                        print("CLIP Score:", clip_res)
                # tqdm
                t.update()

        eval_info_dict = dict()
        torch.cuda.empty_cache()

        # fid
        if compute_fid:
            eval_info_dict["fid"] = fid_stats.compute_fid()
        # clip-score
        if self.cfg.compute_clip_score:
            eval_info_dict["clip_score"] = clip_score_stats.compute()
        return eval_info_dict

    def evaluate(self, step: int, model: BaseT2IModel, f_log=sys.stdout) -> dict[Any, Any]:
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
        self.autoencoder = self.autoencoder.to("cpu")
        return eval_info_dict

    def get_trainable_module_list(self, model: BaseT2IModel) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def load_fsdp_model_to_eval(self, model_to_eval: nn.Module, eval_checkpoint_path: str):
        model_to_eval.cfg.pretrained_path = eval_checkpoint_path
        model_to_eval.cfg.pretrained_source = "dc-ae-fsdp"
        model_to_eval.load_model()

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
            prompts = batch["captions"]
            text_embed_info = self.text_encoder.get_text_embed_info(prompts, self.device)

        if self.cfg.adaptive_latent_channels is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                latent_channels = batch["latent_channels"]
                loss_dict_list, info_list = zip(
                    *[
                        self.model(
                            x=batch["images"][:, :latent_channel],
                            text_embed_info=text_embed_info,
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
                    x=batch["images"],
                    text_embed_info=text_embed_info,
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
