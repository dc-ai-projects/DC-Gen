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
import hashlib
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional, TypeVar

import ipdb
import numpy as np
import pandas
import torch
import torch.nn as nn
import wandb
from omegaconf import MISSING
from torch._dynamo.eval_frame import OptimizedModule
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.api import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import (
    ShardedGradScaler,  # https://github.com/pytorch/pytorch/issues/76607
)
from tqdm import tqdm

from ...models.utils.network import freeze_weights, get_dtype_from_str
from ..data_provider.dc_base import BaseDataProvider
from ..data_provider.dc_mixture import MixtureDataProvider
from ..utils.dist import (
    broadcast_object,
    destroy_process_group,
    dist_barrier,
    dist_init,
    gather_object,
    get_dist_local_rank,
    get_dist_rank,
    get_dist_size,
    is_dist_initialized,
    is_master,
)
from ..utils.ema import EMA, EMA_fsdp
from ..utils.lr import ConstantLRwithWarmup, WSDCosineLR
from ..utils.metric import AverageMeter

__all__ = ["BaseTrainerConfig", "BaseTrainer"]


@dataclass
class BaseTrainerConfig:
    mode: str = "eval"

    # env
    device: str = "cuda"
    seed: int = 0
    amp: str = "fp32"  # "bf16"
    allow_tf32: bool = False
    timeout: Optional[float] = None  # seconds

    # run
    run_dir: str = MISSING
    eval_dir_name: Optional[str] = None
    num_save_samples: int = 64
    save_samples_at_all_ranks: bool = False
    save_all_samples: bool = False

    # process data
    convert_to_amp_dtype_before_forward: bool = False

    # model
    distributed_method: str = "DDP"
    activation_checkpointing_mode: Optional[str] = None
    torch_compile: bool = False
    ema_decay: Any = None
    ema_warmup_steps: int = 2000
    static_graph: bool = True
    find_unused_parameters: bool = False

    # optimizer
    optimizer: str = "AdamW"
    lr: Any = 6.4e-5
    weight_decay: float = 0.0
    no_wd_keys: tuple[str, ...] = ("norm.weight", "bias")
    betas: Any = (0.9, 0.999)

    # lr_scheduler
    lr_scheduler: Any = "ConstantLRwithWarmup"
    warmup_steps: Any = 50000
    warmup_lr: Any = 6.4e-6
    stable_steps: Optional[int] = None
    # training
    checkpoint_dir: str = "${.run_dir}"
    resume: bool = True
    resume_schedule: bool = True
    resume_path: Optional[str] = None
    max_steps: Optional[int] = None
    clip_grad: Optional[float] = None
    save_samples_steps: Optional[int] = 1000
    save_checkpoint_steps: int = 1000
    eval_steps: int = 20000
    save_eval_checkpoint_steps: int = "${.eval_steps}"
    metric_for_best_model: Optional[str] = None
    greater_is_better: Optional[bool] = None
    gradient_accumulation_steps: Optional[int] = None

    # log
    log: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: Optional[str] = None

    # debug
    verbose: bool = False
    skip_training: bool = False


class BaseTrainer:
    def __init__(self, cfg: BaseTrainerConfig):
        self.cfg = cfg
        self.setup_dist_env()
        self.setup_seed()

        if self.cfg.mode in ["eval", "train"]:
            self.eval_data_providers = self.build_eval_data_providers()
            self.setup_model()
            self.setup_run_dir()

        if self.cfg.mode in ["train"]:
            self.train_data_provider = self.build_train_data_provider()
            self.setup_model_for_training()
            self.setup_optimizer()
            self.setup_lr_scheduler()
            self.setup_run_dir_for_training()
            self.setup_logger()
            if cfg.metric_for_best_model is not None and is_master():
                assert cfg.greater_is_better is not None
                if cfg.greater_is_better:
                    self.best_metric = -float("inf")
                else:
                    self.best_metric = float("inf")
            self.global_step = 0
            self.train_sync_generator_cpu = torch.Generator(device=torch.device("cpu"))
            self.train_sync_generator_cpu.manual_seed(cfg.seed)
            self.train_async_generator_cpu = torch.Generator(device=torch.device("cpu"))
            self.train_async_generator_cpu.manual_seed(cfg.seed + self.rank)
            self.train_sync_generator_gpu = torch.Generator(device=torch.device("cuda"))
            self.train_sync_generator_gpu.manual_seed(cfg.seed)
            self.train_async_generator_gpu = torch.Generator(device=torch.device("cuda"))
            self.train_async_generator_gpu.manual_seed(cfg.seed + self.rank)
            if cfg.resume:
                self.try_resume_from_checkpoint()
            if self.cfg.activation_checkpointing_mode is not None:
                if self.cfg.distributed_method == "DDP":
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        self.model.module.enable_activation_checkpointing(self.cfg.activation_checkpointing_mode)
                    else:
                        self.model.enable_activation_checkpointing(self.cfg.activation_checkpointing_mode)
                elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                    self.model.enable_activation_checkpointing(self.cfg.activation_checkpointing_mode)
                else:
                    raise NotImplementedError(f"distributed method {self.cfg.distributed_method} is not supported")
            if self.cfg.torch_compile:
                self.torch_compile()
                if self.cfg.distributed_method == "DDP":
                    torch._dynamo.config.optimize_ddp = False  # https://github.com/pytorch/pytorch/issues/104674

    def setup_dist_env(self) -> None:
        dist_init(timeout=timedelta(seconds=self.cfg.timeout) if self.cfg.timeout is not None else None)
        self.rank, self.local_rank, self.dist_size = get_dist_rank(), get_dist_local_rank(), get_dist_size()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(self.cfg.device)
        if self.cfg.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def setup_seed(self) -> None:
        seed = self.rank + self.cfg.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def build_eval_data_providers(self) -> list[BaseDataProvider]:
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def setup_model(self) -> None:
        self.model = self.build_model().to(self.device)

    def setup_run_dir(self) -> None:
        if is_master():
            os.makedirs(self.cfg.run_dir, exist_ok=True)
            with open(os.path.join(self.cfg.run_dir, "model.txt"), "w") as f:
                f.write(f"{self.network}")
        dist_barrier()

    @property
    def enable_amp(self) -> bool:
        return self.cfg.amp != "fp32"

    @property
    def amp_dtype(self) -> torch.dtype:
        return get_dtype_from_str(self.cfg.amp)

    @property
    def enable_grad_scaler(self) -> bool:
        if self.cfg.distributed_method in ["DDP", "FSDP"]:
            return self.enable_amp
        elif self.cfg.distributed_method == "FSDPWrap":
            return False
        else:
            raise NotImplementedError(f"distributed method {self.cfg.distributed_method} is not support")

    @torch.no_grad()
    def evaluate(self, step: int, model: nn.Module, f_log=sys.stdout) -> dict[str, Any]:
        raise NotImplementedError

    def build_train_data_provider(self) -> BaseDataProvider:
        raise NotImplementedError

    def setup_model_for_training(self):
        if self.cfg.ema_decay is not None:
            if self.cfg.distributed_method == "DDP":
                self.ema = EMA(self.network, self.cfg.ema_decay, self.cfg.ema_warmup_steps)
            elif self.cfg.distributed_method == "FSDP":
                shadows = {}
                ema_decay_list = [self.cfg.ema_decay] if isinstance(self.cfg.ema_decay, float) else self.cfg.ema_decay
                for ema_decay in ema_decay_list:
                    model_copy = copy.deepcopy(self.model)
                    freeze_weights(model_copy)
                    device_mesh = init_device_mesh("cuda", (self.dist_size,))
                    shadows[ema_decay] = FullyShardedDataParallel(
                        model_copy,
                        use_orig_params=True,
                        device_mesh=device_mesh,
                        mixed_precision=MixedPrecision(param_dtype=self.amp_dtype, reduce_dtype=torch.float32),
                    )
                self.ema = EMA_fsdp(shadows)
            elif self.cfg.distributed_method == "FSDPWrap":
                shadows = {}
                ema_decay_list = [self.cfg.ema_decay] if isinstance(self.cfg.ema_decay, float) else self.cfg.ema_decay
                for ema_decay in ema_decay_list:
                    model_copy = copy.deepcopy(self.model)
                    freeze_weights(model_copy)
                    from .fsdp_wrap import fsdp_wrap

                    shadows[ema_decay] = fsdp_wrap(model_copy)
                self.ema = EMA_fsdp(shadows)
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        if is_dist_initialized():
            if self.cfg.distributed_method == "DDP":
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    static_graph=self.cfg.static_graph,
                    find_unused_parameters=self.cfg.find_unused_parameters,
                )
            elif self.cfg.distributed_method == "FSDP":
                # torch._dynamo.config.capture_scalar_outputs = True
                device_mesh = init_device_mesh("cuda", (self.dist_size,))
                self.model = FullyShardedDataParallel(
                    self.model,
                    use_orig_params=True,
                    device_mesh=device_mesh,
                    mixed_precision=MixedPrecision(param_dtype=self.amp_dtype, reduce_dtype=torch.float32),
                )
            elif self.cfg.distributed_method == "FSDPWrap":
                from .fsdp_wrap import fsdp_wrap

                self.model = fsdp_wrap(self.model)
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

    @property
    def network(self) -> nn.Module:
        network = self.model
        if isinstance(network, OptimizedModule):
            network = network._orig_mod
        if isinstance(network, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
            network = network.module
        return network

    def get_trainable_module_list(self, model: nn.Module) -> nn.ModuleList:
        raise NotImplementedError

    def setup_optimizer(self):
        trainable_module_list = self.get_trainable_module_list(self.network)
        no_wd_keys = self.cfg.no_wd_keys

        self.optimizers: list[torch.optim.Optimizer] = []
        for i, trainable_module in enumerate(trainable_module_list):
            lr = self.cfg.lr if isinstance(self.cfg.lr, float) else self.cfg.lr[i]
            betas = self.cfg.betas[i] if isinstance(self.cfg.betas[0], list) else self.cfg.betas
            weight_decay = (
                self.cfg.weight_decay if isinstance(self.cfg.weight_decay, float) else self.cfg.weight_decay[i]
            )
            optimizer_name = self.cfg.optimizer if isinstance(self.cfg.optimizer, str) else self.cfg.optimizer[i]
            param_dict = {}
            for name, param in trainable_module.named_parameters():
                if not param.requires_grad:
                    continue
                opt_config = [weight_decay, lr]
                if any(key in name for key in no_wd_keys):
                    opt_config[0] = 0.0
                opt_key = json.dumps(opt_config)
                param_dict[opt_key] = param_dict.get(opt_key, []) + [(param, name)]

            net_params = []
            for opt_key, param_list in param_dict.items():
                weight_decay, lr = json.loads(opt_key)
                params = [param[0] for param in param_list]
                param_names = [param[1] for param in param_list]
                net_params.append(
                    {"params": params, "weight_decay": weight_decay, "lr": lr, "param_names": param_names}
                )

            if optimizer_name == "AdamW":
                optimizer = torch.optim.AdamW(net_params, lr=lr, betas=betas)
            elif optimizer_name == "Lion":
                from lion_pytorch import Lion

                optimizer = Lion(net_params, lr=lr, betas=betas, decoupled_weight_decay=True)
            elif optimizer_name == "CAMEWrapper":
                from .optimizer import CAMEWrapper

                optimizer = CAMEWrapper(params=net_params, lr=lr, betas=betas, weight_decay=weight_decay)
            else:
                raise ValueError(f"optimizer {optimizer_name} is not supported")

            self.optimizers.append(optimizer)

        if self.enable_grad_scaler:
            if self.cfg.distributed_method == "DDP":
                self.scalers = [torch.GradScaler() for _ in range(len(self.optimizers))]
            elif self.cfg.distributed_method == "FSDP":
                self.scalers = [ShardedGradScaler() for _ in range(len(self.optimizers))]
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

    def setup_lr_scheduler(self):
        self.lr_schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        for i, optimizer in enumerate(self.optimizers):
            lr_scheduler_name = (
                self.cfg.lr_scheduler if isinstance(self.cfg.lr_scheduler, str) else self.cfg.lr_scheduler[i]
            )
            if lr_scheduler_name == "CosineAnnealingLR":
                assert self.cfg.max_steps is not None
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.cfg.max_steps, eta_min=0.0
                )
            elif lr_scheduler_name == "ConstantLR":
                lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            elif lr_scheduler_name == "ConstantLRwithWarmup":
                warmup_lr = self.cfg.warmup_lr if isinstance(self.cfg.warmup_lr, float) else self.cfg.warmup_lr[i]
                warmup_steps = (
                    self.cfg.warmup_steps if isinstance(self.cfg.warmup_steps, int) else self.cfg.warmup_steps[i]
                )
                lr_scheduler = ConstantLRwithWarmup(optimizer, warmup_steps, warmup_lr)
            elif lr_scheduler_name == "WSDCosineLR":
                warmup_lr = self.cfg.warmup_lr if isinstance(self.cfg.warmup_lr, float) else self.cfg.warmup_lr[i]
                warmup_steps = (
                    self.cfg.warmup_steps if isinstance(self.cfg.warmup_steps, int) else self.cfg.warmup_steps[i]
                )
                lr_scheduler = WSDCosineLR(
                    optimizer, warmup_steps, warmup_lr, self.cfg.stable_steps, self.cfg.max_steps
                )
            else:
                raise ValueError(f"lr_scheduler {lr_scheduler} is not supported")
            self.lr_schedulers.append(lr_scheduler)

    def setup_run_dir_for_training(self) -> None:
        if is_master():
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            params_dict = {}
            for optimizer_index, optimizer in enumerate(self.optimizers):
                for group in optimizer.param_groups:
                    lr = group["lr"]
                    wd = group["weight_decay"]
                    for param, name in zip(group["params"], group["param_names"]):
                        param: torch.nn.Parameter
                        params_dict[name] = {
                            "optimizer": optimizer_index,
                            "lr": lr,
                            "wd": wd,
                            "shape": list(param.shape),
                            "mean": f"{param.mean().item():.2E}",
                            "std": f"{param.std().item():.2E}",
                            "first_several_values": ", ".join(f"{value:.4f}" for value in param.view(-1)[:10].tolist()),
                        }
            pandas.DataFrame.from_dict(params_dict, orient="index").to_csv(os.path.join(self.cfg.run_dir, "params.csv"))
            with open(os.path.join(self.cfg.run_dir, "model_training.txt"), "w") as f:
                f.write(f"{self.model}")
        self.latest_file_path = os.path.join(self.cfg.checkpoint_dir, "latest.txt")
        if is_master() and self.cfg.save_samples_steps is not None:
            self.train_save_samples_dir = os.path.join(self.cfg.run_dir, "train_save_samples")
            os.makedirs(self.train_save_samples_dir, exist_ok=True)
        dist_barrier()

    def setup_logger(self):
        if is_master():
            self.train_log = open(os.path.join(self.cfg.run_dir, "log.txt"), "a")
        else:
            self.train_log = sys.stdout
        self.print_and_train_log(f"run_dir: {self.cfg.run_dir}\n", flush=True)

        self.log_to_wandb = self.cfg.log and (self.cfg.wandb_entity is not None or self.cfg.wandb_project is not None)
        if not self.log_to_wandb or not is_master():
            return
        self.logger = wandb.init(
            dir=self.cfg.run_dir,
            entity=self.cfg.wandb_entity,
            project=self.cfg.wandb_project,
            config=vars(self.cfg),
            name=self.cfg.run_dir,
            id=hashlib.sha1(self.cfg.run_dir.encode("utf-8")).hexdigest(),
            resume="allow",
        )
        self.log_dicts = []

    def print_and_train_log(self, message: str, flush: bool = False):
        if is_master():
            print(message, end="", flush=flush)
            self.train_log.write(message)
            if flush:
                self.train_log.flush()

    def cache_log(self, log_dict: dict[str, Any]):
        if not self.log_to_wandb or not is_master():
            return
        if len(self.log_dicts) > 0 and self.log_dicts[-1][0] == self.global_step:
            self.log_dicts[-1][1].update(log_dict)
        else:
            self.log_dicts.append((self.global_step, log_dict))

    def log(self, max_step: Optional[int] = None):
        if not self.log_to_wandb or not is_master():
            return
        for step, log_dict in self.log_dicts:
            if max_step is None or step <= max_step:
                self.logger.log(log_dict, step=step, commit=True)
        self.log_dicts = [
            (step, log_dict) for (step, log_dict) in self.log_dicts if max_step is not None and step > max_step
        ]

    def get_random_states(self) -> list | dict[str, Any]:
        random_state = random.getstate()
        numpy_random_state = np.random.get_state()
        numpy_random_state = (
            numpy_random_state[0],
            numpy_random_state[1].tolist(),
            numpy_random_state[2],
            numpy_random_state[3],
            numpy_random_state[4],
        )
        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state()
        train_sync_generator_cpu_state = self.train_sync_generator_cpu.get_state()
        train_async_generator_cpu_state = self.train_async_generator_cpu.get_state()
        train_sync_generator_gpu_state = self.train_sync_generator_gpu.get_state()
        train_async_generator_gpu_state = self.train_async_generator_gpu.get_state()
        random_states_single_rank = {
            "random_state": random_state,
            "numpy_random_state": numpy_random_state,
            "torch_rng_state": torch_rng_state,
            "torch_cuda_rng_state": torch_cuda_rng_state,
            "train_sync_generator_cpu_state": train_sync_generator_cpu_state,
            "train_async_generator_cpu_state": train_async_generator_cpu_state,
            "train_sync_generator_gpu_state": train_sync_generator_gpu_state,
            "train_async_generator_gpu_state": train_async_generator_gpu_state,
        }
        if self.cfg.distributed_method == "DDP":
            random_states = gather_object(random_states_single_rank)
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            random_states = {f"rank_{self.rank}": random_states_single_rank}
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        return random_states

    def set_random_states(self, random_states: list | dict[str, Any]):
        if self.cfg.distributed_method == "DDP":
            random_states_single_rank = random_states[self.rank]
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            random_states_single_rank = random_states[f"rank_{self.rank}"]
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        random.setstate(random_states_single_rank["random_state"])
        np.random.set_state(random_states_single_rank["numpy_random_state"])
        torch.set_rng_state(random_states_single_rank["torch_rng_state"])
        self.print_and_train_log(f"torch rng state loaded\n")
        torch.cuda.set_rng_state(random_states_single_rank["torch_cuda_rng_state"])
        self.print_and_train_log(f"torch cuda rng state loaded\n")
        if "train_sync_generator_cpu_state" in random_states_single_rank:
            self.train_sync_generator_cpu.set_state(random_states_single_rank["train_sync_generator_cpu_state"])
            self.print_and_train_log(f"train sync generator cpu state loaded\n")
        if "train_async_generator_cpu_state" in random_states_single_rank:
            self.train_async_generator_cpu.set_state(random_states_single_rank["train_async_generator_cpu_state"])
            self.print_and_train_log(f"train async generator cpu state loaded\n")
        if "train_sync_generator_gpu_state" in random_states_single_rank:
            self.train_sync_generator_gpu.set_state(random_states_single_rank["train_sync_generator_gpu_state"])
            self.print_and_train_log(f"train sync generator gpu state loaded\n")
        if "train_async_generator_gpu_state" in random_states_single_rank:
            self.train_async_generator_gpu.set_state(random_states_single_rank["train_async_generator_gpu_state"])
            self.print_and_train_log(f"train async generator gpu state loaded\n")

    def get_train_data_provider_states(self, place_holder: bool = False) -> dict[str, Any]:
        train_data_provider_states = {}
        if isinstance(self.train_data_provider, MixtureDataProvider):
            train_data_provider_states["sampler_state_dict"] = self.train_data_provider.sampler.state_dict(
                self.global_step * self.train_data_provider.cfg.batch_size, place_holder=place_holder
            )
        return train_data_provider_states

    def set_train_data_provider_states(self, train_data_provider_states: dict[str, Any]):
        if isinstance(self.train_data_provider, MixtureDataProvider):
            self.train_data_provider.sampler.load_state_dict(train_data_provider_states["sampler_state_dict"])

    def get_train_states(self, only_model_state_dict: bool = False, place_holder: bool = False) -> dict[str, Any]:
        train_states = {}

        train_states["global_step"] = self.global_step
        train_states["dist_size"] = self.dist_size
        if self.cfg.metric_for_best_model is not None and is_master():
            train_states["best_metric"] = self.best_metric

        if self.cfg.ema_decay is not None:
            if self.cfg.distributed_method == "DDP":
                train_states["ema"] = (
                    {
                        decay: self.get_trainable_module_list(shadow).state_dict()
                        for decay, shadow in self.ema.shadows.items()
                    }
                    if is_master()
                    else None
                )
            elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                train_states["ema"] = self.ema.state_dict()
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

        if only_model_state_dict:
            if self.cfg.distributed_method == "DDP":
                train_states["model_state_dict"] = (
                    self.get_trainable_module_list(self.network).state_dict() if is_master() else None
                )
            elif self.cfg.distributed_method == "FSDP":
                train_states["model_state_dict"] = get_model_state_dict(self.model)
            elif self.cfg.distributed_method == "FSDPWrap":
                train_states["model_state_dict"] = get_model_state_dict(
                    self.model, options=StateDictOptions(cpu_offload=True)
                )
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
            return train_states

        if self.cfg.distributed_method == "DDP":
            if is_master():
                train_states["model_state_dict"] = self.get_trainable_module_list(self.network).state_dict()
            else:
                train_states["model_state_dict"] = None
            for i, optimizer in enumerate(self.optimizers):
                train_states[f"optimizer_state_dict_{i}"] = optimizer.state_dict()
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            train_states["model_state_dict"], train_states["optimizer_state_dict"] = get_state_dict(
                self.model, self.optimizers
            )
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        if self.enable_grad_scaler:
            for i, scaler in enumerate(self.scalers):
                train_states[f"scaler_{i}"] = scaler.state_dict()
        train_states.update(self.get_train_data_provider_states(place_holder))
        for i, lr_scheduler in enumerate(self.lr_schedulers):
            train_states[f"lr_scheduler_{i}"] = lr_scheduler.state_dict()
        train_states["random_states"] = self.get_random_states()
        return train_states

    def model_name_to_model_file_name(self, model_name: str) -> str:
        if self.cfg.distributed_method == "DDP":
            return f"{model_name}.pt"
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            return f"{model_name}"
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

    def save_model(self, model_name: str, only_model_state_dict: bool = False):
        train_states = self.get_train_states(only_model_state_dict)
        model_path = os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name(model_name))
        if model_name == "checkpoint":  # avoid partially saved checkpoints
            model_path_ = os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name("checkpoint_"))
            if self.cfg.distributed_method == "DDP":
                if is_master():
                    torch.save(train_states, model_path_)
            elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                if is_master():
                    shutil.rmtree(model_path_, ignore_errors=True)
                dist_barrier()
                torch.distributed.checkpoint.save(train_states, checkpoint_id=model_path_)
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
            dist_barrier()
            if is_master():
                with open(self.latest_file_path, "w") as f:
                    f.write("checkpoint_")
                if os.path.isdir(model_path_):
                    shutil.rmtree(model_path, ignore_errors=True)
                    shutil.copytree(model_path_, model_path)
                else:
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    shutil.copy(model_path_, model_path)
                with open(self.latest_file_path, "w") as f:
                    f.write("checkpoint")
                if os.path.isdir(model_path_):
                    shutil.rmtree(model_path_, ignore_errors=True)
                else:
                    os.remove(model_path_)
        else:
            if self.cfg.distributed_method == "DDP":
                if is_master():
                    torch.save(train_states, model_path)
            elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                tmp_model_path = os.path.join(self.cfg.checkpoint_dir, "tmp")
                if is_master():
                    shutil.rmtree(tmp_model_path, ignore_errors=True)
                dist_barrier()
                torch.distributed.checkpoint.save(train_states, checkpoint_id=tmp_model_path)
                dist_barrier()
                if is_master():
                    dcp_to_torch_save(tmp_model_path, model_path + ".pt")
                    shutil.rmtree(tmp_model_path, ignore_errors=True)
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        dist_barrier()
        self.print_and_train_log(f"save model to {model_path} at step {self.global_step}\n", flush=True)

    def resume_schedule(self, train_states: dict[str, Any]):
        self.set_train_data_provider_states(train_states)
        for i, lr_scheduler in enumerate(self.lr_schedulers):
            lr_scheduler.load_state_dict(train_states[f"lr_scheduler_{i}"])
            self.print_and_train_log(f"lr_scheduler {i} loaded\n")
        if self.dist_size == train_states["dist_size"]:
            self.set_random_states(train_states["random_states"])
        else:
            self.print_and_train_log(
                f"Warning: failed to load random_states. Current dist_size: {self.dist_size}. Loaded dist_size: {train_states['dist_size']}\n"
            )
        self.global_step = train_states["global_step"]
        self.print_and_train_log(f"global_step={self.global_step}\n")
        if self.cfg.metric_for_best_model is not None and is_master():
            if "best_metric" in train_states:
                self.best_metric = train_states["best_metric"]
            self.print_and_train_log(f"best {self.cfg.metric_for_best_model}={self.best_metric:.2f}\n")

    def resume_from_checkpoint(self, checkpoint_path: str, resume_source: str):
        self.print_and_train_log(f"loading checkpoint {checkpoint_path}\n")

        if self.cfg.distributed_method == "DDP":
            train_states = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.get_trainable_module_list(self.network).load_state_dict(train_states["model_state_dict"])
            self.print_and_train_log(f"model loaded\n")
            for i, optimizer in enumerate(self.optimizers):
                optimizer_state = train_states[f"optimizer_state_dict_{i}"]
                optimizer.load_state_dict(optimizer_state)
                self.print_and_train_log(f"optimizer {i} loaded\n")
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            train_states = self.get_train_states(place_holder=True)
            torch.distributed.checkpoint.load(train_states, checkpoint_id=checkpoint_path)
            set_state_dict(
                self.model,
                self.optimizers,
                model_state_dict=train_states["model_state_dict"],
                optim_state_dict=train_states["optimizer_state_dict"],
                options=StateDictOptions(strict=True),
            )
            self.print_and_train_log(f"model and optimizers loaded\n")
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

        if self.cfg.ema_decay is not None:
            if self.cfg.distributed_method == "DDP":
                for decay in self.ema.shadows:
                    self.get_trainable_module_list(self.ema.shadows[decay]).load_state_dict(train_states["ema"][decay])
            elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                self.ema.load_state_dict(train_states["ema"])
            else:
                raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
            self.print_and_train_log(f"ema state_dict loaded\n")

        if self.enable_grad_scaler:
            for i, scaler in enumerate(self.scalers):
                scaler.load_state_dict(train_states[f"scaler_{i}"])
                self.print_and_train_log(f"scaler {i} loaded\n")

        if self.cfg.resume_schedule or resume_source == "checkpoint":
            self.resume_schedule(train_states)

        self.print_and_train_log(f"checkpoint {checkpoint_path} loaded\n", flush=True)

    def try_resume_from_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name("checkpoint"))
        checkpoint_path_ = os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name("checkpoint_"))

        if (
            not os.path.exists(checkpoint_path)
            and not os.path.exists(checkpoint_path_)
            and self.cfg.resume_path is not None
        ):
            resume_source = "resume_path"
            dist_barrier()
            if is_master():
                if os.path.isdir(self.cfg.resume_path):
                    shutil.copytree(self.cfg.resume_path, checkpoint_path)
                else:
                    shutil.copy(self.cfg.resume_path, checkpoint_path)
                with open(self.latest_file_path, "w") as f:
                    f.write("checkpoint")
            dist_barrier()
        else:
            resume_source = "checkpoint"

        if os.path.exists(checkpoint_path) or os.path.exists(checkpoint_path_):
            with open(self.latest_file_path, "r") as f:
                model_name = f.read()
            self.resume_from_checkpoint(
                os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name(model_name)), resume_source
            )
        else:
            self.print_and_train_log("can not find a checkpoint, will train from scratch\n")

    def torch_compile(self):
        self.model = torch.compile(self.model)

    def load_fsdp_model_to_eval(self, model_to_eval: nn.Module, eval_checkpoint_path: str):
        raise NotImplementedError

    def get_model_to_eval(self) -> nn.Module:
        if self.cfg.distributed_method == "DDP":
            if self.cfg.ema_decay is not None:
                model_to_eval = next(iter(self.ema.shadows.values()))
            else:
                model_to_eval = self.network
        elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
            eval_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"step_{self.global_step}.pt")
            model_to_eval = self.build_model()
            try:
                self.load_fsdp_model_to_eval(model_to_eval, eval_checkpoint_path)
            except:
                dist_barrier()
                if is_master():
                    with open(self.latest_file_path, "r") as f:
                        model_name = f.read()
                    dcp_to_torch_save(
                        os.path.join(self.cfg.checkpoint_dir, self.model_name_to_model_file_name(model_name)),
                        eval_checkpoint_path,
                    )
                dist_barrier()
                self.load_fsdp_model_to_eval(model_to_eval, eval_checkpoint_path)
            model_to_eval = model_to_eval.cuda()
        else:
            raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")
        return model_to_eval

    def check_best_metric(self, eval_info_dict: dict[str, Any]) -> bool:
        if not is_master():
            return False
        if self.cfg.metric_for_best_model is None:
            return False
        keys = self.cfg.metric_for_best_model.split(".")
        current_metric = eval_info_dict
        self.print_and_train_log(f"keys: {keys}\n")
        self.print_and_train_log(f"current_metric: {current_metric}\n")
        for key in keys:
            if key not in current_metric:
                return False
            current_metric = current_metric[key]
        self.print_and_train_log(f"current_metric: {current_metric}\n")
        assert isinstance(current_metric, float)
        if (self.cfg.greater_is_better and current_metric > self.best_metric) or (
            not self.cfg.greater_is_better and current_metric < self.best_metric
        ):
            self.best_metric = current_metric
            return True
        else:
            return False

    def init_train_data_provider(self):
        if isinstance(self.train_data_provider, MixtureDataProvider):  # for data loaders with infinite length
            self.data_loader_length = None
        else:  # for conventional data loaders
            self.data_loader_length = len(self.train_data_provider.data_loader)
        self.print_and_train_log(f"data_loader_length: {self.data_loader_length}\n")
        if self.data_loader_length is not None:
            self.epoch = self.global_step // self.data_loader_length
            self.train_data_provider.sampler.set_epoch(self.epoch)
            self.train_data_provider.set_batch_index(self.global_step - self.data_loader_length * self.epoch)
        else:
            self.epoch = None
        self.data_loader_iter = iter(self.train_data_provider.data_loader)

    def get_next_train_batch(self) -> dict[str, Any]:
        try:
            batch = next(self.data_loader_iter)
        except StopIteration:
            # for conventional data loaders
            assert self.data_loader_length is not None and self.epoch is not None
            self.epoch += 1
            self.train_data_provider.sampler.set_epoch(self.epoch)
            self.train_data_provider.set_batch_index(0)
            self.data_loader_iter = iter(self.train_data_provider.data_loader)
            batch = next(self.data_loader_iter)
        return batch

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        def recursive_to_dtype_and_device(data):
            if isinstance(data, dict):
                return {key: recursive_to_dtype_and_device(value) for key, value in data.items()}
            elif isinstance(data, list):
                return [recursive_to_dtype_and_device(item) for item in data]
            elif isinstance(data, torch.Tensor):
                data = data.to(device=self.device)
                if self.cfg.convert_to_amp_dtype_before_forward and torch.is_floating_point(data):
                    data = data.to(dtype=self.amp_dtype)
                return data
            else:
                return data

        batch = recursive_to_dtype_and_device(batch)
        return batch

    def get_batch_size(self, batch: dict[str, Any]) -> int:
        raise NotImplementedError

    def split_batch_for_gradient_accumulation(self, batch: dict[str, Any], batch_size: int) -> list[dict[str, Any]]:
        assert (
            self.cfg.gradient_accumulation_steps is not None and batch_size % self.cfg.gradient_accumulation_steps == 0
        )
        batch_chunk_size = batch_size // self.cfg.gradient_accumulation_steps
        batch_chunk_list: list[dict[str, Any]] = []
        for batch_chunk_index in range(self.cfg.gradient_accumulation_steps):
            batch_chunk_start = batch_chunk_index * batch_chunk_size
            batch_chunk_end = batch_chunk_start + batch_chunk_size
            batch_chunk_list.append({})
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    assert value.shape[0] == batch_size
                    batch_chunk_list[batch_chunk_index][key] = value[batch_chunk_start:batch_chunk_end]
                elif isinstance(value, list):
                    assert len(value) % self.cfg.gradient_accumulation_steps == 0
                    batch_chunk_list[batch_chunk_index][key] = value[batch_chunk_start:batch_chunk_end]
                elif value is None:
                    batch_chunk_list[batch_chunk_index][key] = None
                else:
                    ipdb.set_trace()
                    raise ValueError(
                        f"Value type {type(value)} not supported for split_batch_for_gradient_accumulation"
                    )
        return batch_chunk_list

    def model_forward(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        raise NotImplementedError

    def merge_output_for_gradient_accumulation(
        self, loss_dict_chunks: list[dict[int, torch.Tensor]], info_chunks: list[dict[str, Any]], batch_size: int
    ) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        assert self.cfg.gradient_accumulation_steps is not None
        loss_dict = {}
        for key in loss_dict_chunks[0]:
            loss_dict[key] = torch.mean(torch.stack([loss_dict_chunk[key] for loss_dict_chunk in loss_dict_chunks]))
        batch_chunk_size = batch_size // self.cfg.gradient_accumulation_steps

        T = TypeVar("T")

        def merge_info_recursive(info_chunks: list[T]) -> T:
            assert all(type(info_chunk) == type(info_chunks[0]) for info_chunk in info_chunks[1:])
            if isinstance(info_chunks[0], dict):
                assert all(info_chunk.keys() == info_chunks[0].keys() for info_chunk in info_chunks[1:])
                info = {}
                for key in info_chunks[0].keys():
                    info[key] = merge_info_recursive([info_chunk[key] for info_chunk in info_chunks])
            elif isinstance(info_chunks[0], list):
                assert all(len(info_chunk) == batch_chunk_size for info_chunk in info_chunks)
                info = sum(info_chunks, start=[])
            elif isinstance(info_chunks[0], torch.Tensor):
                if all(info_chunk.shape[0] == batch_chunk_size for info_chunk in info_chunks):
                    info = torch.cat(info_chunks, dim=0)
                elif all(info_chunk.ndim == 0 for info_chunk in info_chunks):
                    info = torch.mean(torch.stack(info_chunks))
                else:
                    raise ValueError(f"tensor shape {[info_chunk.shape for info_chunk in info_chunks]} not supported")
            elif isinstance(info_chunks[0], float):
                info = sum(info_chunks) / len(info_chunks)
            else:
                raise ValueError(f"info type {type(info_chunks[0])} not supported")
            return info

        info = merge_info_recursive(info_chunks)
        return loss_dict, info

    def train_step(self, batch: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
        """
        The implementation of gradient accumulation is slightly different from normal forward and backward.
        In normal forward and backward, we can support multiple loss functions with overlapping parameters because we can call self.model.zero_grad() function after each backward and parameter update.
        However, after enabling gradient accumulation, in order to accumulate gradients, we can only call self.model.zero_grad() function at the end. In this case, multiple loss functions with overlapping parameters may cause abnormal behavior.
        """
        if self.cfg.gradient_accumulation_steps is None:
            loss_dict, info = self.model_forward(batch)

            for optimizer_idx, loss in loss_dict.items():
                optimizer = self.optimizers[optimizer_idx]
                lr_scheduler = self.lr_schedulers[optimizer_idx]
                if self.enable_grad_scaler:
                    scaler = self.scalers[optimizer_idx]
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                # gradient clip
                if self.cfg.clip_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        sum([param_group["params"] for param_group in optimizer.param_groups], start=[]),
                        self.cfg.clip_grad,
                    )
                else:
                    grad_norm = torch.norm(
                        torch.tensor(
                            [
                                torch.norm(p.grad)
                                for param_group in optimizer.param_groups
                                for p in param_group["params"]
                                if p.grad is not None
                            ]
                        )
                    )
                info[f"grad_norm_{optimizer_idx}"] = grad_norm.item()
                # step
                if self.enable_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                self.model.zero_grad()
        else:
            batch_size = self.get_batch_size(batch)
            batch_chunk_list = self.split_batch_for_gradient_accumulation(batch, batch_size)

            loss_dict_chunks: list[dict[int, torch.Tensor]] = []
            info_chunks: list[dict[str, Any]] = []
            for batch_chunk_index, batch_chunk in enumerate(batch_chunk_list):
                if batch_chunk_index != self.cfg.gradient_accumulation_steps - 1:
                    if self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                        context_func = self.model.no_sync
                    elif self.cfg.distributed_method == "DDP":
                        context_func = nullcontext
                    else:
                        raise NotImplementedError(f"distributed method {self.cfg.distributed_method} is not supported")
                else:
                    context_func = nullcontext
                with context_func():
                    with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=True):
                        loss_dict_chunk, info_chunk = self.model_forward(batch_chunk)
                    loss_dict_chunks.append(loss_dict_chunk)
                    info_chunks.append(info_chunk)
                    for optimizer_idx, loss in loss_dict_chunk.items():
                        scaled_loss = 1 / self.cfg.gradient_accumulation_steps * loss
                        optimizer = self.optimizers[optimizer_idx]
                        lr_scheduler = self.lr_schedulers[optimizer_idx]
                        if self.enable_grad_scaler:
                            scaler = self.scalers[optimizer_idx]
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()

            loss_dict, info = self.merge_output_for_gradient_accumulation(loss_dict_chunks, info_chunks, batch_size)

            for optimizer_idx in loss_dict:
                optimizer = self.optimizers[optimizer_idx]
                lr_scheduler = self.lr_schedulers[optimizer_idx]
                if self.enable_grad_scaler:
                    scaler = self.scalers[optimizer_idx]
                    scaler.unscale_(optimizer)
                # gradient clip
                if self.cfg.clip_grad is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        sum([param_group["params"] for param_group in optimizer.param_groups], start=[]),
                        self.cfg.clip_grad,
                    )
                else:
                    grad_norm = torch.norm(
                        torch.tensor(
                            [
                                torch.norm(p.grad)
                                for param_group in optimizer.param_groups
                                for p in param_group["params"]
                                if p.grad is not None
                            ]
                        )
                    )
                info[f"grad_norm_{optimizer_idx}"] = grad_norm.item()
                # step
                if self.enable_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
            self.model.zero_grad()

        if self.cfg.ema_decay is not None:
            self.ema.step(self.network, self.global_step)

        return loss_dict, info

    def print_verbose_info(self, batch: dict[str, Any], loss_dict: dict[str, Any], info: dict[str, Any]):
        raise NotImplementedError

    def save_samples(self, batch: dict[str, Any], info: dict[str, Any]):
        raise NotImplementedError

    def get_current_step_train_loss_dict(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError

    def after_step(
        self,
        batch: dict[str, Any],
        loss_dict: dict[str, Any],
        info: dict[str, Any],
        average_loss_dict: dict[str, AverageMeter],
        log_dict: dict[str, Any],
        postfix_dict: dict[str, Any],
    ) -> None:
        if self.cfg.verbose:
            self.print_verbose_info(batch, loss_dict, info)

        if self.cfg.save_samples_steps is not None and (is_master() or self.cfg.save_all_samples):
            if self.global_step % self.cfg.save_samples_steps == 0:
                self.save_samples(batch, info)

        postfix_dict["bs"] = batch_size = self.get_batch_size(batch)
        current_step_train_loss_dict = self.get_current_step_train_loss_dict(loss_dict, info)
        for loss_key, loss_value in current_step_train_loss_dict.items():
            assert isinstance(loss_value, float), f"{loss_key} {loss_value} is not a float"
            if loss_key not in average_loss_dict:
                average_loss_dict[loss_key] = AverageMeter(is_distributed=is_dist_initialized())
            average_loss_dict[loss_key].update(loss_value, batch_size)
            log_dict[f"train/{loss_key}"] = loss_value
        postfix_dict["global_step"] = self.global_step
        for i, lr_scheduler in enumerate(self.lr_schedulers):
            lr = lr_scheduler.get_last_lr()[0]
            postfix_dict[f"lr_{i}"] = lr
            log_dict[f"train/lr_{i}"] = lr
            if f"grad_norm_{i}" in info:
                postfix_dict[f"grad_norm_{i}"] = info[f"grad_norm_{i}"]
                log_dict[f"train/grad_norm_{i}"] = info[f"grad_norm_{i}"]
        for loss_key in average_loss_dict:
            postfix_dict[loss_key] = average_loss_dict[loss_key].avg

    def check_termination(self, loss_dict: dict[str, Any], info: dict[str, Any]) -> bool:
        return False

    def train(self) -> None:
        self.model.train()
        average_loss_dict: dict[str, AverageMeter] = dict()

        with tqdm(
            total=None,
            desc="Train",
            disable=not is_master(),
            file=self.train_log,
            mininterval=10.0,
        ) as t:
            last_step_time = time.time()
            self.init_train_data_provider()
            while True:
                if self.global_step > 0 and self.global_step % self.cfg.eval_steps == 0:
                    model_to_eval = self.get_model_to_eval()

                    eval_info_dict = self.evaluate(self.global_step, model=model_to_eval, f_log=self.train_log)
                    del model_to_eval
                    if len(eval_info_dict) == 1:
                        eval_info_dict = next(iter(eval_info_dict.values()))
                    self.print_and_train_log(f"eval info dict: {eval_info_dict}\n", flush=True)

                    dist_barrier()
                    if self.cfg.distributed_method == "DDP":
                        if (
                            self.global_step % self.cfg.save_eval_checkpoint_steps == 0
                            and is_master()
                            and os.path.exists(self.latest_file_path)
                        ):
                            with open(self.latest_file_path, "r") as f:
                                model_name = f.read()
                            model_path = os.path.join(
                                self.cfg.checkpoint_dir, self.model_name_to_model_file_name(model_name)
                            )
                            eval_model_path = os.path.join(
                                self.cfg.checkpoint_dir,
                                self.model_name_to_model_file_name(f"step_{self.global_step}"),
                            )
                            shutil.copy(model_path, eval_model_path)
                        dist_barrier()
                    elif self.cfg.distributed_method in ["FSDP", "FSDPWrap"]:
                        eval_checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"step_{self.global_step}.pt")
                        if (
                            self.global_step % self.cfg.save_eval_checkpoint_steps != 0
                            and is_master()
                            and os.path.exists(eval_checkpoint_path)
                        ):
                            os.remove(eval_checkpoint_path)
                    else:
                        raise ValueError(f"distributed_method {self.cfg.distributed_method} is not supported")

                    is_best_metric = self.check_best_metric(eval_info_dict)
                    is_best_metric = broadcast_object(is_best_metric)
                    if is_best_metric:
                        self.save_model(f"best", only_model_state_dict=True)

                    self.cache_log({"eval/" + key: value for key, value in eval_info_dict.items()})
                    self.log()
                    self.model.train()

                if self.cfg.max_steps is not None and self.global_step >= self.cfg.max_steps:
                    self.print_and_train_log(
                        f"max steps {self.cfg.max_steps} reached, terminate training\n", flush=True
                    )
                    break

                self.global_step += 1
                # get data
                batch = self.get_next_train_batch()

                if self.cfg.skip_training:  # for debugging data provider
                    t.update()
                    continue

                step_start_time = time.time()
                log_dict: dict[str, Any] = dict()
                postfix_dict: dict[str, Any] = dict()
                log_dict["train/load_data_time"] = step_start_time - last_step_time
                batch = self.prepare_batch(batch)
                loss_dict, info = self.train_step(batch)
                self.after_step(batch, loss_dict, info, average_loss_dict, log_dict, postfix_dict)
                t.set_postfix(postfix_dict, refresh=False)
                t.update()

                step_end_time = time.time()
                log_dict["train/step_time"] = step_end_time - step_start_time

                self.cache_log(log_dict)
                if self.global_step % self.cfg.save_checkpoint_steps == 0:
                    self.save_model("checkpoint")
                    self.log(max_step=self.global_step - 1 if self.global_step % self.cfg.eval_steps == 0 else None)

                if self.check_termination(loss_dict, info):
                    break

                last_step_time = time.time()

    def run_eval(self):
        start_time = time.time()
        eval_info_dict = self.evaluate(0, self.model)
        eval_info_dict["evaluation_time"] = time.time() - start_time
        if is_master():
            for key, value in eval_info_dict.items():
                print(f"{key}: {value}")

    def run(self):
        if self.cfg.mode == "eval":
            self.run_eval()
        elif self.cfg.mode == "train":
            self.train()
        else:
            raise ValueError(f"trainer mode {self.cfg.mode} is not supported")

    def __del__(self):
        if is_master() and self.cfg.verbose:
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
        if self.cfg.mode == "train" and is_master():
            self.train_log.close()
        destroy_process_group()
