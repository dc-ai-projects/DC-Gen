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

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel

from .dist import dist_barrier, get_dist_rank, is_master
from .dtype import get_dtype_from_str

__all__ = ["EMA"]


class EMA:
    def __init__(
        self,
        model: nn.Module,
        decay: float | list[float],
        distributed_method: str,
        warmup_steps: int = 2000,
        device: str = "cuda",
        dtype: str = "fp32",
    ):
        super().__init__()
        self.rank = get_dist_rank()

        if isinstance(decay, float):
            decay_list = [decay]
        else:
            decay_list = decay
        self.decay_list: list[float] = decay_list
        self.distributed_method = distributed_method
        self.warmup_steps = warmup_steps
        self.device = torch.device(device)
        self.dtype = get_dtype_from_str(dtype)

        self.shadows: dict[float, dict[str, torch.Tensor]] = {}
        for decay in self.decay_list:
            shadow: dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    shadow[name] = param.detach().clone().to(dtype=self.dtype, device=self.device)
            for name, buffer in model.named_buffers():
                if buffer.is_floating_point:
                    shadow[name] = buffer.detach().clone().to(dtype=self.dtype, device=self.device)
            self.shadows[decay] = shadow

        self.original = None

    @torch.no_grad()
    def step(self, model: nn.Module, global_step: int):
        for decay in self.decay_list:
            if self.warmup_steps == 0:
                decay_this_step = decay
            else:
                decay_this_step = decay * (1 - math.exp(-global_step / self.warmup_steps))
            shadow = self.shadows[decay]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    shadow[name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")].mul_(
                        decay_this_step
                    ).add_(param.detach().to(dtype=self.dtype, device=self.device), alpha=1 - decay_this_step)
            for name, buffer in model.named_buffers():
                if buffer.is_floating_point:
                    shadow[name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")].mul_(
                        decay_this_step
                    ).add_(buffer.detach().to(dtype=self.dtype, device=self.device), alpha=1 - decay_this_step)

    @torch.no_grad()
    def store(self, model: nn.Module):
        # save all model weight
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.detach().clone().to(self.device)
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point:
                self.original[name] = buffer.detach().clone().to(self.device)

    @torch.no_grad()
    def copy_to(self, model: nn.Module, decay: Optional[float] = None):
        if self.distributed_method == "DDP":
            pass
        elif self.distributed_method in ["FSDP", "FSDPWrap"]:
            # before copy param back to original model, ensure clear their shard cache
            # this is related to issue: https://github.com/pytorch/pytorch/issues/117421
            with FullyShardedDataParallel.summon_full_params(
                model, writeback=False, rank0_only=True, offload_to_cpu=True
            ):
                pass
        else:
            raise ValueError(f"distributed method {self.distributed_method} is not supported")

        if decay is None:
            shadow = self.shadows[self.decay_list[0]]
        else:
            shadow = self.shadows[decay]

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(
                    shadow[name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")].to(
                        device=param.device, dtype=param.dtype
                    )
                )
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point:
                buffer.copy_(
                    shadow[name.replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")].to(
                        device=buffer.device, dtype=buffer.dtype
                    )
                )

    @torch.no_grad()
    def restore(self, model: nn.Module):
        # restore all model weight
        if self.original is None:
            raise RuntimeError("Must call `store()` before `restore()`.")

        if self.distributed_method == "DDP":
            pass
        elif self.distributed_method in ["FSDP", "FSDPWrap"]:
            # before copy param back to original model, ensure clear their shard cache
            # this is related to issue: https://github.com/pytorch/pytorch/issues/117421
            with FullyShardedDataParallel.summon_full_params(
                model, writeback=False, rank0_only=True, offload_to_cpu=True
            ):
                pass
        else:
            raise ValueError(f"distributed method {self.distributed_method} is not supported")

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.copy_(self.original[name].to(dtype=param.dtype, device=param.device))
        for name, buffer in model.named_buffers():
            if buffer.is_floating_point:
                buffer.copy_(self.original[name].to(dtype=buffer.dtype, device=buffer.device))
        self.original = None

    def load_state_dict(self, state_dict):
        if self.distributed_method == "DDP":
            pass
        elif self.distributed_method in ["FSDP", "FSDPWrap"]:
            state_dict = state_dict[f"rank_{self.rank}"]
        else:
            raise ValueError(f"distributed method {self.distributed_method} is not supported")
        for decay in self.decay_list:
            shadow = self.shadows[decay]
            for key in shadow:
                shadow[key].copy_(state_dict[decay][key])

    def state_dict(self):
        if self.distributed_method == "DDP":
            return self.shadows
        elif self.distributed_method in ["FSDP", "FSDPWrap"]:
            return {f"rank_{self.rank}": self.shadows}
        else:
            raise ValueError(f"distributed method {self.distributed_method} is not supported")

    def get_ema_model_state_dict(self, model: nn.Module):
        """
        Get full model state dict with EMA, including parameters requiring grad that are stored in EMA, and parameters not requiring grad that are stored in model.
        """
        self.store(model)

        ema_model_state_dict = {}
        for decay in self.decay_list:
            # if we want to get full parameter with ema, we should first write to model
            self.copy_to(model, decay=decay)

            if self.distributed_method == "DDP":
                orig_model_state_dict = model.state_dict()
                model_state_dict = {}
                for key, value in orig_model_state_dict.items():
                    model_state_dict[key.removeprefix("_orig_mod.").removeprefix("module.")] = (
                        value.detach().clone().float().cpu()
                    )
            elif self.distributed_method in ["FSDP", "FSDPWrap"]:
                model_state_dict = {}
                with FullyShardedDataParallel.summon_full_params(
                    model, writeback=False, rank0_only=True, offload_to_cpu=True
                ):
                    for name, param in model.named_parameters():
                        model_state_dict[
                            name.replace("_checkpoint_wrapped_module.", "")
                            .replace("_orig_mod.", "")
                            .replace("_fsdp_wrapped_module.", "")
                        ] = (param.detach().clone().float().cpu())
                    for name, buffer in model.named_buffers():
                        model_state_dict[
                            name.replace("_checkpoint_wrapped_module.", "")
                            .replace("_orig_mod.", "")
                            .replace("_fsdp_wrapped_module.", "")
                        ] = (buffer.detach().clone().float().cpu())
            else:
                raise ValueError(f"distributed method {self.distributed_method} is not supported")

            if is_master():
                ema_model_state_dict[decay] = model_state_dict

        self.restore(model)
        return ema_model_state_dict
