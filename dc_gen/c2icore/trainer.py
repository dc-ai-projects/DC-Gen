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
from typing import Any, Optional

import numpy as np
import pandas
import torch
from omegaconf import MISSING
from PIL import Image
from torch import nn
from tqdm import tqdm

from ..apps.metrics.cmmd.cmmd import CMMDStats, CMMDStatsConfig
from ..apps.metrics.fid.fid import FIDStats, FIDStatsConfig
from ..apps.metrics.inception_score.inception_score import InceptionScoreStats, InceptionScoreStatsConfig
from ..apps.trainer.dc_trainer import BaseTrainer, BaseTrainerConfig
from ..apps.utils.dist import broadcast_object, dist_barrier, is_dist_initialized, is_master, sync_tensor
from ..apps.utils.metric import AverageMeter
from ..c2i_model_zoo import REGISTERED_DCAE_DIFFUSION_MODEL, DCAE_Diffusion_HF
from ..models.utils.network import get_dtype_from_str, get_params_num
from .autoencoder import Autoencoder, AutoencoderConfig
from .data_provider.base import C2ICoreDataProvider
from .data_provider.latent_imagenet import LatentImageNetDataProvider, LatentImageNetDataProviderConfig
from .data_provider.sample_class import SampleClassDataProvider, SampleClassDataProviderConfig
from .models.base import BaseC2IModel, BaseC2IModelConfig

__all__ = ["C2ICoreTrainerConfig", "C2ICoreTrainer"]


@dataclass
class C2ICoreTrainerConfig(BaseTrainerConfig):
    # env
    allow_tf32: bool = True

    resolution: int = 512
    cfg_scale: Any = 1.0
    search_optimal_cfg_scale: bool = False
    cfg_scale_candidates: tuple[float, ...] = (
        1.0,
        1.05,
        1.1,
        1.15,
        1.2,
        1.25,
        1.3,
        1.4,
        1.5,
        1.6,
        1.7,
        1.8,
        1.9,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
    )
    save_image_format: str = "jpg"
    save_latent_samples: bool = False
    latent_samples_dir: Optional[str] = None

    # eval data providers
    eval_data_provider: str = "sample_class"
    sample_class: SampleClassDataProviderConfig = field(default_factory=SampleClassDataProviderConfig)

    # autoencoder
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    autoencoder_dtype: str = "fp32"
    eval_autoencoder_setting_list: Optional[tuple[int]] = None

    # model
    model: str = MISSING
    print_nfe: bool = False

    # metrics
    compute_fid: bool = True
    fid: FIDStatsConfig = field(default_factory=FIDStatsConfig)
    compute_inception_score: bool = True
    inception_score: InceptionScoreStatsConfig = field(default_factory=InceptionScoreStatsConfig)
    compute_cmmd: bool = False
    cmmd: CMMDStatsConfig = field(default_factory=CMMDStatsConfig)

    # train data providers
    train_data_provider: str = "latent_imagenet"
    train_data_scaling_factor: Any = None
    latent_imagenet: LatentImageNetDataProviderConfig = field(default_factory=LatentImageNetDataProviderConfig)

    # process data
    adaptive_latent_channels: Optional[tuple[int]] = None
    num_sample_latent_channels: int = 1
    always_sample_max_channel: bool = False

    # training
    save_eval_checkpoint_steps: int = 100000


class C2ICoreTrainer(BaseTrainer):
    def __init__(self, cfg: C2ICoreTrainerConfig):
        super().__init__(cfg)
        self.cfg: C2ICoreTrainerConfig

        if cfg.mode == "train" and cfg.compute_fid:
            assert os.path.exists(self.cfg.fid.ref_path), f"fid ref {self.cfg.fid.ref_path} not found"
            if self.cfg.fid.precision_recall_ref_path is not None:
                assert os.path.exists(self.cfg.fid.precision_recall_ref_path)
        if cfg.mode == "train" and cfg.compute_cmmd:
            assert os.path.exists(self.cfg.cmmd.ref_path), f"cmmd ref {self.cfg.cmmd.ref_path} not found"

    def build_eval_data_providers(self) -> list[C2ICoreDataProvider]:
        if self.cfg.eval_data_provider == "sample_class":
            return [SampleClassDataProvider(self.cfg.sample_class)]
        else:
            raise ValueError(f"eval data provider {self.cfg.eval_data_provider} is not supported")

    def get_possible_models(self) -> dict[str, tuple[BaseC2IModelConfig, type[BaseC2IModel]]]:
        raise NotImplementedError

    def build_model(self) -> BaseC2IModel:
        possible_models = self.get_possible_models()

        if self.cfg.model in possible_models:
            model_cfg, model_class = possible_models[self.cfg.model]
            model_cfg.input_size = self.cfg.resolution // self.autoencoder.spatial_compression_ratio
            model = model_class(copy.deepcopy(model_cfg))
        elif self.cfg.model in REGISTERED_DCAE_DIFFUSION_MODEL:
            model = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/{self.cfg.model}").diffusion_model
        else:
            raise ValueError(f"model {self.cfg.model} is not supported among {possible_models.keys()}")

        if is_master():
            print(f"params: {get_params_num(model):.2f} M")

        return model

    def setup_model(self) -> None:
        if self.cfg.model in REGISTERED_DCAE_DIFFUSION_MODEL:
            autoencoder = DCAE_Diffusion_HF.from_pretrained(f"mit-han-lab/{self.cfg.model}").autoencoder
        else:
            autoencoder = Autoencoder(self.cfg.autoencoder)
        self.autoencoder = autoencoder.to(device=self.device, dtype=get_dtype_from_str(self.cfg.autoencoder_dtype))
        super().setup_model()

    def load_latent_samples(self) -> torch.Tensor:
        idx = self.rank
        all_latent_samples = []
        while True:
            latent_samples_path = os.path.join(self.cfg.latent_samples_dir, f"latent_samples_{idx}.pt")
            if os.path.exists(latent_samples_path):
                all_latent_samples.append(torch.load(latent_samples_path, weights_only=True))
            else:
                break
            idx += self.dist_size
        all_latent_samples = torch.cat(all_latent_samples, dim=0)
        return all_latent_samples

    @torch.no_grad()
    def evaluate_single(
        self,
        step: int,
        network: nn.Module,
        f_log=sys.stdout,
        cfg_scale: float = 1.0,
        autoencoder_setting_index: int = 0,
        additional_dir_name: Optional[str] = None,
    ) -> dict[str, Any]:
        # channels
        latent_channels = self.autoencoder.model_list[autoencoder_setting_index].latent_channels

        network.eval()
        eval_generator = torch.Generator(device=torch.device("cuda"))
        eval_generator.manual_seed(self.cfg.seed + self.rank)

        data_loader = self.eval_data_providers[0].data_loader

        # metrics
        if self.cfg.compute_fid:
            fid_stats = FIDStats(self.cfg.fid)
        if self.cfg.compute_inception_score:
            inception_score_stats = InceptionScoreStats(self.cfg.inception_score)
        if self.cfg.compute_cmmd:
            cmmd_stats = CMMDStats(self.cfg.cmmd)

        if self.cfg.eval_dir_name is not None:
            evaluate_dir = os.path.join(self.cfg.run_dir, self.cfg.eval_dir_name, additional_dir_name)
        else:
            evaluate_dir = os.path.join(self.cfg.run_dir, f"images/{step}", additional_dir_name)
        if is_master():
            os.makedirs(evaluate_dir, exist_ok=True)
        if is_dist_initialized():
            dist_barrier()

        def update_metrics(image_samples_uint8):
            if self.cfg.compute_fid:
                fid_stats.add_data(image_samples_uint8)
            if self.cfg.compute_inception_score:
                inception_score_stats.add_data(image_samples_uint8)
            if self.cfg.compute_cmmd:
                cmmd_stats.add_data(image_samples_uint8)

        if self.cfg.latent_samples_dir is not None:
            all_latent_samples = self.load_latent_samples()
            for start in tqdm(range(0, all_latent_samples.shape[0], data_loader.batch_size), disable=not is_master()):
                latent_samples = all_latent_samples[
                    start : min(start + data_loader.batch_size, all_latent_samples.shape[0]), :latent_channels
                ].cuda()
                image_samples = self.autoencoder.decode(latent_samples, autoencoder_setting_index)
                assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
                update_metrics(image_samples_uint8)
        else:
            if self.cfg.save_latent_samples:
                all_latent_samples = []
            with tqdm(
                total=len(data_loader),
                desc="Valid Step #{}".format(step),
                disable=not is_master(),
                file=f_log,
                mininterval=10.0,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as t:
                num_saved_samples = 0
                for _, (inputs, inputs_null) in enumerate(data_loader):
                    # preprocessing
                    inputs = inputs.cuda()
                    inputs_null = inputs_null.cuda()
                    # sample
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                        latent_samples = network.generate(
                            inputs, inputs_null, cfg_scale, eval_generator, latent_channels
                        )
                    if self.cfg.save_latent_samples:
                        all_latent_samples.append(latent_samples.cpu())
                    image_samples = self.autoencoder.decode(
                        latent_samples.to(get_dtype_from_str(self.cfg.autoencoder_dtype)), autoencoder_setting_index
                    )
                    # assert torch.isnan(image_samples).sum() == 0, "NaN detected!"
                    image_samples_uint8 = torch.clamp(127.5 * image_samples + 128.0, 0, 255).to(dtype=torch.uint8)
                    image_samples_numpy = image_samples_uint8.permute(0, 2, 3, 1).cpu().numpy()
                    if self.cfg.print_nfe:
                        print(f"rank={self.rank}, nfe={network.nfe}")
                        network.nfe = 0

                    if (
                        num_saved_samples < self.cfg.num_save_samples
                        and (is_master() or self.cfg.save_samples_at_all_ranks)
                    ) or self.cfg.save_all_samples:
                        image_samples_PIL = [Image.fromarray(image) for image in image_samples_numpy]
                        for j, image_sample_PIL in enumerate(image_samples_PIL):
                            if self.cfg.save_all_samples:
                                idx = num_saved_samples * self.dist_size + self.rank
                            else:
                                if num_saved_samples >= self.cfg.num_save_samples:
                                    break
                                idx = num_saved_samples
                            image_sample_PIL.save(
                                os.path.join(
                                    evaluate_dir,
                                    f"{self.rank}_{idx:05d}_{inputs[j].item()}.{self.cfg.save_image_format}",
                                )
                            )
                            num_saved_samples += 1
                        del image_samples_PIL

                    update_metrics(image_samples_uint8)
                    ## tqdm
                    t.update()

        if self.cfg.save_latent_samples:
            all_latent_samples = torch.cat(all_latent_samples, dim=0)
            torch.save(all_latent_samples, os.path.join(evaluate_dir, f"latent_samples_{self.rank}.pt"))

        eval_info_dict = dict()
        torch.cuda.empty_cache()
        # fid
        if self.cfg.compute_fid:
            eval_info_dict["fid"] = fid_stats.compute_fid()
            if self.cfg.fid.precision_recall_ref_path is not None:
                eval_info_dict.update(fid_stats.compute_precision_recall())
        # compute_inception_score
        if self.cfg.compute_inception_score:
            eval_info_dict.update(inception_score_stats.compute())
        # cmmd
        if self.cfg.compute_cmmd:
            eval_info_dict["cmmd"] = cmmd_stats.compute_cmmd()

        return eval_info_dict

    def evaluate(
        self, step: int, model: nn.Module, f_log=sys.stdout, cfg_scale: Optional[float] = None
    ) -> dict[Any, Any]:
        results_path = os.path.join(self.cfg.run_dir, "eval_results.csv")
        if os.path.exists(results_path):
            results = pandas.read_csv(results_path, index_col=0)
        else:
            results = pandas.DataFrame()
        eval_info_dict = {}
        autoencoder_setting_list = (
            self.cfg.eval_autoencoder_setting_list
            if self.cfg.eval_autoencoder_setting_list is not None
            else list(range(self.cfg.autoencoder.num_settings))
        )
        if cfg_scale is None:
            cfg_scale_list = self.cfg.cfg_scale if isinstance(self.cfg.cfg_scale, list) else [self.cfg.cfg_scale]
        else:
            cfg_scale_list = [cfg_scale]
        for autoencoder_setting_index in autoencoder_setting_list:
            for cfg_scale in cfg_scale_list:
                eval_dir_name = f"{self.cfg.eval_dir_name}" if self.cfg.eval_dir_name is not None else f"step_{step}"
                setting = ""
                setting += f"autoencoder_setting_{autoencoder_setting_index}"
                setting += f"_cfg_{cfg_scale}"
                index = f"{eval_dir_name}_{setting}"
                if index in results.index:
                    eval_info_dict[setting] = results.loc[[index]].to_dict(orient="index")[index]
                else:
                    eval_info_dict[setting] = self.evaluate_single(
                        step,
                        model,
                        f_log,
                        cfg_scale,
                        autoencoder_setting_index=autoencoder_setting_index,
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
                    dist_barrier()
        return eval_info_dict

    def search_optimal_cfg_scale(self):
        l_index, r_index = 0, len(self.cfg.cfg_scale_candidates)
        cfg_scale_1_eval_info_dict = next(
            iter(self.evaluate(0, self.model, cfg_scale=self.cfg.cfg_scale_candidates[l_index]).values())
        )
        l_fid = broadcast_object(cfg_scale_1_eval_info_dict["fid"])
        r_fid = float("inf")
        eval_info_dicts = {1.0: cfg_scale_1_eval_info_dict}
        while r_index - l_index >= 3:
            mid_1_index = l_index + (r_index - l_index) // 3
            mid_2_index = l_index + 2 * (r_index - l_index) // 3
            if is_master():
                print(
                    f"rank={self.rank}, l_index={l_index}, r_index={r_index}, mid_1_index={mid_1_index}, mid_2_index={mid_2_index}"
                )
                print(f"eval cfg scale: {self.cfg.cfg_scale_candidates[mid_1_index]}")
            self.setup_seed()
            mid_1_eval_info_dict = next(
                iter(self.evaluate(0, self.model, cfg_scale=self.cfg.cfg_scale_candidates[mid_1_index]).values())
            )
            mid_1_fid = broadcast_object(mid_1_eval_info_dict["fid"])
            eval_info_dicts[self.cfg.cfg_scale_candidates[mid_1_index]] = mid_1_eval_info_dict
            if is_master():
                print(f"eval cfg scale: {self.cfg.cfg_scale_candidates[mid_2_index]}")
            self.setup_seed()
            mid_2_eval_info_dict = next(
                iter(self.evaluate(0, self.model, cfg_scale=self.cfg.cfg_scale_candidates[mid_2_index]).values())
            )
            mid_2_fid = broadcast_object(mid_2_eval_info_dict["fid"])
            eval_info_dicts[self.cfg.cfg_scale_candidates[mid_2_index]] = mid_2_eval_info_dict
            if mid_1_fid < mid_2_fid:
                r_index = mid_2_index
                r_fid = mid_2_fid
            else:
                l_index = mid_1_index
                l_fid = mid_1_fid
            dist_barrier()

        assert l_index + 2 == r_index
        mid_index = l_index + 1
        self.setup_seed()
        cfg_scale_best_eval_info_dict = next(
            iter(self.evaluate(0, self.model, cfg_scale=self.cfg.cfg_scale_candidates[mid_index]).values())
        )
        eval_info_dicts[self.cfg.cfg_scale_candidates[mid_index]] = cfg_scale_best_eval_info_dict
        if is_master():
            for cfg_scale in sorted(eval_info_dicts):
                print(f"cfg scale {cfg_scale}: {eval_info_dicts[cfg_scale]}")
        if is_master():
            print(f"optimal cfg scale {self.cfg.cfg_scale_candidates[mid_index]}: {cfg_scale_best_eval_info_dict}")

    def run_eval(self):
        if self.cfg.search_optimal_cfg_scale:
            self.search_optimal_cfg_scale()
        else:
            super().run_eval()

    def build_train_data_provider(self) -> C2ICoreDataProvider:
        if self.cfg.train_data_provider == "latent_imagenet":
            train_data_provider = LatentImageNetDataProvider(self.cfg.latent_imagenet)
        else:
            raise NotImplementedError(f"train dataset {self.cfg.train_data_provider} is not supported")

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

    def get_trainable_module_list(self, model: BaseC2IModel) -> nn.ModuleList:
        return model.get_trainable_modules_list()

    def load_fsdp_model_to_eval(self, model_to_eval: BaseC2IModel, eval_checkpoint_path: str):
        model_to_eval.cfg.pretrained_path = eval_checkpoint_path
        model_to_eval.cfg.pretrained_source = "dc-ae-fsdp"
        model_to_eval.load_model()

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = super().prepare_batch(batch)

        if self.train_data_scaling_factor is None:
            pass
        elif isinstance(self.train_data_scaling_factor, float):
            batch["images"] = batch["images"] * self.train_data_scaling_factor
        elif isinstance(self.train_data_scaling_factor, torch.Tensor):
            batch["images"] = (
                batch["images"] * self.train_data_scaling_factor[None, : batch["images"].shape[1], None, None]
            )
        else:
            raise ValueError(f"train_data_scaling_factor {self.train_data_scaling_factor} is not supported")

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
        images = batch["images"]
        labels = batch["labels"]
        projections = batch["projections"] if "projections" in batch else None
        latent_channels = batch["latent_channels"]

        if self.cfg.adaptive_latent_channels is not None:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                loss_dict_list, info_list = zip(
                    *[
                        self.model(
                            images[:, :latent_channel], labels, self.train_async_generator_gpu, projections=projections
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
                loss_dict, info = self.model(images, labels, self.train_async_generator_gpu, projections=projections)

        if "detailed_loss_dict" in info:
            for key in info["detailed_loss_dict"]:
                info["detailed_loss_dict"][key] = info["detailed_loss_dict"][key].item()

        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        if is_master():
            print(
                f"global step {self.global_step}, images {batch['images'].sum()}, loss {loss_dict[0].item()}, grad_norm {info['grad_norm_0']}",
                flush=True,
            )

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
        postfix_dict["shape"] = list(batch["images"].shape)

    def check_termination(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> bool:
        mean_loss = sync_tensor(loss_dict[0], reduce="mean").item()
        if np.isnan(mean_loss):
            self.print_and_train_log(f"NaN detected, terminate training")
            return True
        return super().check_termination(loss_dict, info)
