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
from ..apps.metrics.clip_score import CLIPScoreStats
from ..apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.trainer.fsdp_wrap import fsdp_wrap
from ..apps.utils.config import get_config
from ..apps.utils.dist import dist_barrier, is_dist_initialized, is_master, sync_tensor
from ..apps.utils.dtype import convert_to_dtype_recursive, get_dtype_from_str
from ..apps.utils.metric import AverageMeter
from ..models.utils.network import freeze_weights, get_params_num
from .data_provider.base_eval import T2ICoreEvalDataProvider
from .data_provider.image_mixture import T2ICoreImageMixtureDataProvider, T2ICoreImageMixtureDataProviderConfig
from .data_provider.latent_mixture import T2ICoreLatentMixtureDataProvider, T2ICoreLatentMixtureDataProviderConfig
from .data_provider.mjhq_text_prompt import MJHQTextPromptDataProvider, MJHQTextPromptDataProviderConfig
from .models.base import BaseT2IModel, BaseT2IModelConfig
from .models.flux import Flux, FluxConfig
from .models.sana_sprint import SanaSprint, SanaSprintConfig
from .models.sana_t2i import SanaT2I, SanaT2IConfig
from .models.zimage import ZImage, ZImageConfig
from .text_encoder import T2ICoreTextEncoder

__all__ = ["T2ICoreTrainerConfig", "T2ICoreTrainer"]


@dataclass
class T2ICoreTrainerConfig(BaseTrainerConfig):
    # env
    resolution: int = 512
    save_image_format: str = "png"
    cfg_scale: float = 4.5
    pag_scale: float = 1.0

    # text encoder
    text_encoders: tuple[str, ...] = ()

    # eval data providers
    eval_data_providers: tuple[str, ...] = ()
    mjhq_text_prompt: MJHQTextPromptDataProviderConfig = field(
        default_factory=lambda: MJHQTextPromptDataProviderConfig(resolution="${..resolution}")
    )

    # train data providers
    train_data_type: str = "latent"
    train_data_scaling_factor: Any = None
    mixture: T2ICoreLatentMixtureDataProviderConfig = field(
        default_factory=lambda: T2ICoreLatentMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}"
        )
    )
    image_mixture: T2ICoreImageMixtureDataProviderConfig = field(
        default_factory=lambda: T2ICoreImageMixtureDataProviderConfig(
            save_checkpoint_steps="${..save_checkpoint_steps}"
        )
    )

    # autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    autoencoder_dtype: str = "fp32"

    # model
    model: str = MISSING
    model_dtype: Optional[str] = None
    sana_t2i: SanaT2IConfig = field(default_factory=lambda: SanaT2IConfig(text_encoder_id="${..text_encoders.0}"))
    flux: FluxConfig = field(default_factory=FluxConfig)
    zimage: ZImageConfig = field(default_factory=lambda: ZImageConfig(text_encoder_id="${..text_encoders.0}"))
    sana_sprint: SanaSprintConfig = field(
        default_factory=lambda: SanaSprintConfig(text_encoder_id="${..text_encoders.0}")
    )

    # metrics
    compute_fid: bool = True
    compute_clip_score: bool = True
    compute_metrics_with_jpeg: bool = True

    # dc-ae 1.5
    adaptive_latent_channels: Optional[tuple[int]] = None
    num_sample_latent_channels: int = 1
    always_sample_max_channel: bool = False

    # steps
    save_samples_steps: Optional[int] = 500
    save_checkpoint_steps: int = 500
    eval_steps: int = 5000

    # offload
    offload: bool = False


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

    def build_train_data_provider(
        self,
    ) -> T2ICoreLatentMixtureDataProvider | T2ICoreImageMixtureDataProvider:
        if self.cfg.train_data_type == "latent":
            train_data_provider = T2ICoreLatentMixtureDataProvider(self.cfg.mixture)
        elif self.cfg.train_data_type == "image":
            train_data_provider = T2ICoreImageMixtureDataProvider(self.cfg.image_mixture)
        else:
            raise ValueError(f"train_data_type {self.cfg.train_data_type} is not supported")

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
        possible_models = {
            "sana_t2i": (self.cfg.sana_t2i, SanaT2I),
            "flux": (self.cfg.flux, Flux),
            "sana_sprint": (self.cfg.sana_sprint, SanaSprint),
            "zimage": (self.cfg.zimage, ZImage),
        }
        return possible_models

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
        # Offload initialization: keep all components on CPU until their execution stage.
        target_device = "cpu" if self.cfg.offload else self.device

        self.text_encoder = T2ICoreTextEncoder(self.cfg.text_encoders)
        freeze_weights(self.text_encoder)
        self.text_encoder.to(target_device)

        self.autoencoder = Autoencoder(self.cfg.autoencoder).to(dtype=get_dtype_from_str(self.cfg.autoencoder_dtype))
        freeze_weights(self.autoencoder)
        self.autoencoder.to(target_device)

        self.model = self.build_model().to(target_device)
        torch.cuda.empty_cache()

    def setup_model_for_training(self) -> None:
        if self.cfg.offload and (not is_dist_initialized() or self.cfg.distributed_method == "DDP"):
            self.model.to(self.device)

        super().setup_model_for_training()

        if not self.cfg.offload:
            return

        if is_dist_initialized() and self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            self.text_encoder = fsdp_wrap(self.text_encoder)
        else:
            self.text_encoder.to(self.device)

        if self.cfg.train_data_type == "image":
            self.autoencoder.to(self.device)
        elif self.cfg.train_data_type == "latent":
            self.autoencoder.to("cpu")
        else:
            raise ValueError(f"train_data_type {self.cfg.train_data_type} is not supported")

        # Training-ready state: model and text encoder are on GPU.
        # Image training also keeps the autoencoder on GPU.
        torch.cuda.empty_cache()

    def restore_training_component_placement(self) -> None:
        if self.cfg.offload:
            self.model.to(self.device)
            self.text_encoder.to(self.device)

        if self.cfg.train_data_type == "image":
            self.autoencoder.to(self.device)
        elif self.cfg.train_data_type == "latent":
            self.autoencoder.to("cpu")
        else:
            raise ValueError(f"train_data_type {self.cfg.train_data_type} is not supported")

        # Training-ready state: model and text encoder are on GPU.
        # Image training also keeps the autoencoder on GPU.
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(
        self,
        network: BaseT2IModel,
        prompt_list: list[str],
        cfg_scale: float,
        pag_scale: float,
        eval_generator: torch.Generator,
    ) -> torch.Tensor:
        if self.cfg.offload:
            # Text encoding: text encoder on GPU; diffusion model and autoencoder on CPU.
            network.to("cpu")
            self.autoencoder.to("cpu")
            torch.cuda.empty_cache()
            self.text_encoder.to(self.device)

        text_embed_info = self.text_encoder(prompt_list, self.device)
        text_embed_info = convert_to_dtype_recursive(text_embed_info, self.amp_dtype)

        if self.cfg.offload:
            # Denoising: diffusion model on GPU; text encoder and autoencoder on CPU.
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            network.to(self.device)

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
            latent_samples = network(
                "generate",
                text_embed_info=text_embed_info,
                cfg_scale=cfg_scale,
                pag_scale=pag_scale,
                generator=eval_generator,
            )

        if self.cfg.offload:
            # Decoding: autoencoder on GPU; diffusion model and text encoder on CPU.
            network.to("cpu")
            torch.cuda.empty_cache()
            self.autoencoder.to(self.device)

        image_samples = self.autoencoder.decode(latent_samples.to(get_dtype_from_str(self.cfg.autoencoder_dtype)))

        if self.cfg.offload:
            # Post-generation offload: evaluation or inference leaves all components on CPU.
            self.autoencoder.to("cpu")
            torch.cuda.empty_cache()

        return image_samples

    def get_train_data_provider_states(self, place_holder: bool = False) -> dict[str, Any]:
        train_data_provider_states = {}
        train_data_provider_states["sampler_state_dict"] = self.train_data_provider.sampler.state_dict(
            self.global_step, place_holder=place_holder
        )
        return train_data_provider_states

    def set_train_data_provider_states(self, train_data_provider_states: dict[str, Any]):
        self.train_data_provider.sampler.load_state_dict(train_data_provider_states["sampler_state_dict"])

    def _should_save_eval_samples(
        self,
        data_provider: T2ICoreEvalDataProvider,
        num_saved_samples: int,
    ) -> bool:
        if data_provider.cfg.save_all_samples or self.cfg.save_all_samples:
            return True
        return num_saved_samples < self.cfg.num_save_samples and (is_master() or self.cfg.save_samples_at_all_ranks)

    @torch.no_grad()
    def evaluate_single(
        self,
        data_provider: T2ICoreEvalDataProvider,
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

        data_loader = data_provider.data_loader

        # metrics
        compute_fid = self.cfg.compute_fid and data_provider.cfg.fid_ref_path is not None
        if compute_fid:
            assert os.path.exists(os.path.expanduser(data_provider.cfg.fid_ref_path))
            fid_stats = FIDStats(FIDStatsConfig(ref_path=os.path.expanduser(data_provider.cfg.fid_ref_path)))
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
                image_samples = self.generate(
                    network=network,
                    prompt_list=prompts,
                    cfg_scale=cfg_scale,
                    pag_scale=pag_scale,
                    eval_generator=eval_generator,
                )

                # assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
                image_samples_numpy = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()

                if self._should_save_eval_samples(data_provider, num_saved_samples):
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
        if not self.cfg.offload:
            self.autoencoder = self.autoencoder.to(self.device)
        results_path = os.path.join(self.cfg.run_dir, "eval_results.csv")
        if os.path.exists(results_path):
            results = pandas.read_csv(results_path, index_col=0)
        else:
            results = pandas.DataFrame()
        eval_info_dict = {}
        cfg_scale_list = self.cfg.cfg_scale if isinstance(self.cfg.cfg_scale, list) else [self.cfg.cfg_scale]
        for cfg_scale in cfg_scale_list:
            for data_provider_name, data_provider in zip(self.cfg.eval_data_providers, self.eval_data_providers):
                eval_dir_name = f"{self.cfg.eval_dir_name}" if self.cfg.eval_dir_name is not None else f"step_{step}"
                setting = f"{data_provider_name}_cfg_{cfg_scale}"
                index = f"{eval_dir_name}_{setting}"
                if index in results.index:
                    eval_info_dict[setting] = results.loc[[index]].to_dict(orient="index")[index]
                else:
                    eval_info_dict[setting] = self.evaluate_single(
                        data_provider=data_provider,
                        step=step,
                        network=model,
                        f_log=f_log,
                        cfg_scale=cfg_scale,
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

    def get_trainable_module_list(self, model: BaseT2IModel) -> nn.ModuleList:
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

    def _get_train_inputs(self, batch: dict[str, Any]) -> torch.Tensor:
        if self.cfg.train_data_type == "image":
            return batch["image"]
        if self.cfg.train_data_type == "latent":
            return batch["images"]
        raise ValueError(f"train_data_type {self.cfg.train_data_type} is not supported")

    def model_forward(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        with torch.no_grad():
            prompts = batch["captions"]
            text_embed_info = self.text_encoder(prompts, self.device)
            text_embed_info = convert_to_dtype_recursive(text_embed_info, self.amp_dtype)
            train_inputs = self._get_train_inputs(batch)
            if self.cfg.train_data_type == "image":
                model_inputs = self.autoencoder.encode(
                    train_inputs.to(dtype=get_dtype_from_str(self.cfg.autoencoder_dtype))
                ).detach()
            elif self.cfg.train_data_type == "latent":
                model_inputs = train_inputs
            else:
                raise ValueError(f"train_data_type {self.cfg.train_data_type} is not supported")

        if self.cfg.adaptive_latent_channels is not None:
            if self.cfg.model != "zimage":
                raise ValueError(
                    f"adaptive_latent_channels is only supported for model zimage, got model {self.cfg.model}"
                )
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                latent_channels = batch["latent_channels"]
                model_kwargs_list = []
                channel_indices = torch.arange(model_inputs.shape[1], device=model_inputs.device)
                for latent_channel in latent_channels:
                    if latent_channel <= 0 or latent_channel > model_inputs.shape[1]:
                        raise ValueError(
                            f"Z-Image adaptive latent channel count must be in [1, {model_inputs.shape[1]}], "
                            f"got {latent_channel}"
                        )
                    channel_mask = channel_indices[None, :] < latent_channel
                    channel_mask = channel_mask.expand(model_inputs.shape[0], -1)
                    model_kwargs_list.append({"channel_mask": channel_mask})
                loss_dict_list, info_list = zip(
                    *[
                        self.model(
                            "forward_train",
                            x=model_inputs,
                            text_embed_info=text_embed_info,
                            generator=self.train_async_generator_gpu,
                            **model_kwargs,
                        )
                        for model_kwargs in model_kwargs_list
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
                    x=model_inputs,
                    text_embed_info=text_embed_info,
                    generator=self.train_async_generator_gpu,
                )

        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        train_inputs = self._get_train_inputs(batch)
        for i in range(2):
            if self.rank == i:
                print(
                    f"global step {self.global_step}, rank {self.rank}, images {train_inputs.sum()}, loss {loss_dict[0].item()}, grad_norm {info['grad_norm_0']}, memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB",
                    flush=True,
                )
                torch.cuda.reset_peak_memory_stats()
            dist_barrier()

    def save_samples(self, batch: dict[str, Any], info: dict[str, Any]):
        pass

    def get_current_step_train_loss_dict(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
        return info["detailed_loss_dict"]

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        return self._get_train_inputs(batch).shape[0]

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
        postfix_dict["shape"] = self._get_train_inputs(batch).shape

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
    cfg: T2ICoreTrainerConfig = get_config(T2ICoreTrainerConfig)
    trainer = T2ICoreTrainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
