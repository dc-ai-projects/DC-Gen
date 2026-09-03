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
import time
from dataclasses import dataclass

import ipdb
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm

from ...apps.utils.config import get_config
from ...apps.utils.dist import is_master
from ...apps.utils.video import VideoLoader, VideoSizeTransform, write_video
from ...models.utils.network import get_params_num
from ...stae_model_zoo import REGISTERED_DCAEV_MODEL
from ..models.dc_ae_v import DCAEV
from ..models.wan_22_vae import Wan22VAE
from ..trainer import STAECoreTrainer, STAECoreTrainerConfig


@dataclass
class TestEfficiencyConfig(STAECoreTrainerConfig):
    amp: str = "bf16"
    log: bool = False
    run_dir: str = "tmp"

    test_num_params: bool = True
    test_inference_throughput: bool = False
    test_encoder_inference_throughput: bool = False
    test_decoder_inference_throughput: bool = False
    test_training_throughput: bool = False
    test_encoder_blockwise_latency: bool = False
    test_decoder_blockwise_latency: bool = False

    batch_size: int = 1
    resolution: int = 256
    h: int = "${.resolution}"
    w: int = "${.resolution}"
    num_frames: int = 17

    warmup_iterations: int = 5
    iterations: int = 20


@torch.no_grad()
def inference_step(trainer: STAECoreTrainer, cfg: TestEfficiencyConfig, device: torch.device):
    samples = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
        trainer.model.reconstruct_video(samples)


@torch.no_grad()
def encoder_inference_step(trainer: STAECoreTrainer, cfg: TestEfficiencyConfig, device: torch.device):
    samples = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
        y = trainer.model.encode(samples)


@torch.no_grad()
def decoder_inference_step(trainer: STAECoreTrainer, device: torch.device, latent_shape: tuple[int, ...]):
    latent = torch.randn(*latent_shape, dtype=trainer.amp_dtype, device=device)
    with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
        y = trainer.model.decode(latent)


def training_step(trainer: STAECoreTrainer, cfg: TestEfficiencyConfig, device: torch.device):
    samples = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
    trainer.train_step({"videos": samples, "latent_channels": None})


@torch.no_grad()
def test_block_latency(
    block: nn.Module, x: torch.Tensor, device: torch.device, dtype: torch.dtype, warmup_iterations: int, iterations: int
):
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
        for _ in range(warmup_iterations):
            block(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in range(iterations):
            block(x)
        torch.cuda.synchronize()
        end_time = time.time()
    return (end_time - start_time) / iterations


def main():
    cfg: TestEfficiencyConfig = get_config(TestEfficiencyConfig)
    trainer = STAECoreTrainer(cfg)
    device = torch.device("cuda")

    size_transform = VideoSizeTransform("CenterCrop", "DMCrop", "Round")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )
    assert os.path.exists("assets/video/drone.mp4")
    video_loader = VideoLoader("assets/video/drone.mp4")
    frames = size_transform(
        video_loader, video_loader.get_frame_count(), video_loader.get_fps(), h=cfg.h, w=cfg.w, t=cfg.num_frames
    )
    x = torch.stack([transform(frame) for frame in frames], dim=1)[None].to(device=device, dtype=trainer.amp_dtype)

    if cfg.torch_compile:
        trainer.model.encode = torch.compile(trainer.model.encode)
        trainer.model.decode = torch.compile(trainer.model.decode)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
            latent = trainer.model.encode(x)
            y = trainer.model.decode(latent)

    write_video(
        os.path.join(cfg.run_dir, "demo.mp4"),
        y[0].cpu() * 0.5 + 0.5,
        fps=8,
    )

    if cfg.test_num_params and is_master():
        print(f"num params: {get_params_num(trainer.model):.2f}")
        if cfg.model in REGISTERED_DCAEV_MODEL:
            print(f"encoder params: {get_params_num(trainer.model.encoder):.2f}")
            print(f"decoder params: {get_params_num(trainer.model.decoder):.2f}")

    if cfg.test_training_throughput:
        torch.set_grad_enabled(True)
        trainer.model.train()
        for _ in tqdm(range(cfg.warmup_iterations), desc="warm up", disable=not is_master()):
            training_step(trainer, cfg, device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in tqdm(range(cfg.iterations), desc="test", disable=not is_master()):
            training_step(trainer, cfg, device)
        torch.cuda.synchronize()
        end_time = time.time()

        if is_master():
            print(f"latency {(end_time-start_time)/cfg.iterations:.2f} s")
            print(f"throughput {cfg.iterations*cfg.batch_size/(end_time-start_time):.2f} samples/s")
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    if cfg.test_inference_throughput:
        torch.set_grad_enabled(False)
        for _ in tqdm(range(cfg.warmup_iterations), desc="warm up", disable=not is_master()):
            inference_step(trainer, cfg, device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in tqdm(range(cfg.iterations), desc="test", disable=not is_master()):
            inference_step(trainer, cfg, device)
        torch.cuda.synchronize()
        end_time = time.time()

        if is_master():
            print(f"latency {(end_time-start_time)/cfg.iterations:.2f} s")
            print(f"throughput {cfg.iterations*cfg.batch_size/(end_time-start_time):.2f} samples/s")
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    if cfg.test_encoder_inference_throughput:
        torch.set_grad_enabled(False)

        for _ in tqdm(range(cfg.warmup_iterations), desc="warm up", disable=not is_master()):
            encoder_inference_step(trainer, cfg, device)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in tqdm(range(cfg.iterations), desc="test", disable=not is_master()):
            encoder_inference_step(trainer, cfg, device)
        torch.cuda.synchronize()
        end_time = time.time()

        if is_master():
            print(f"latency {(end_time-start_time)/cfg.iterations:.2f} s")
            print(f"throughput {cfg.iterations*cfg.batch_size/(end_time-start_time):.2f} samples/s")
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    if cfg.test_decoder_inference_throughput:
        torch.set_grad_enabled(False)
        samples = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
        with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
            latent = trainer.model.encode(samples)

        for _ in tqdm(range(cfg.warmup_iterations), desc="warm up", disable=not is_master()):
            decoder_inference_step(trainer, device, latent.shape)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        for _ in tqdm(range(cfg.iterations), desc="test", disable=not is_master()):
            decoder_inference_step(trainer, device, latent.shape)
        torch.cuda.synchronize()
        end_time = time.time()

        if is_master():
            print(f"latency {(end_time-start_time)/cfg.iterations:.2f} s")
            print(f"throughput {cfg.iterations*cfg.batch_size/(end_time-start_time):.2f} samples/s")
            print(f"memory {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    if cfg.test_encoder_blockwise_latency:
        torch.set_grad_enabled(False)
        results: dict[str, float] = {}
        if isinstance(trainer.model, DCAEV):
            x = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
            encoder = trainer.model.encoder.to(dtype=trainer.amp_dtype)
            if trainer.model.cfg.num_pad_frames > 0:
                x = F.pad(x, (0, 0, 0, 0, trainer.model.cfg.num_pad_frames, 0), mode="replicate")
            results["encoder.project_in"] = test_block_latency(
                encoder.project_in, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
            x, _ = encoder.project_in(x)
            for stage_id, stage in enumerate(encoder.stages):
                if len(stage.op_list) == 0:
                    continue
                depth = encoder.cfg.depth_list[stage_id]
                for block_id in range(depth):
                    block = stage.op_list[block_id]
                    results[f"encoder.stage_{stage_id}.block_{block_id}"] = test_block_latency(
                        block, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                    )
                    x, _ = block(x)
                if stage_id < encoder.num_stages - 1 and depth > 0:
                    results[f"encoder.stage_{stage_id}.downsample"] = test_block_latency(
                        stage.op_list[-1], x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                    )
                    x, _ = stage.op_list[-1](x)
            results["encoder.project_out"] = test_block_latency(
                encoder.project_out, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
        elif isinstance(trainer.model, Wan22VAE):
            from ..models.wan_22_vae import patchify

            trainer.model = trainer.model.to(dtype=trainer.amp_dtype)
            encoder = trainer.model.model.encoder
            x = x.repeat(cfg.batch_size, 1, 1, 1, 1)
            x = patchify(x, patch_size=2)
            x = F.pad(x, (0, 0, 0, 0, 3, 0), mode="replicate")

            results[f"encoder.conv1"] = test_block_latency(
                encoder.conv1, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
            x = encoder.conv1(x)
            for layer_id, layer in enumerate(encoder.downsamples):
                results[f"encoder.downsamples_{layer_id}"] = test_block_latency(
                    layer, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                )
                x = layer(x)
            for layer_id, layer in enumerate(encoder.middle):
                results[f"encoder.middle_{layer_id}"] = test_block_latency(
                    layer, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                )
                x = layer(x)
            for layer_id, layer in enumerate(encoder.head):
                results[f"encoder.head_{layer_id}"] = test_block_latency(
                    layer, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                )
                x = layer(x)
            results[f"conv1"] = test_block_latency(
                trainer.model.model.conv1, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
            x = trainer.model.model.conv1(x).chunk(2, dim=1)[0]
            y = trainer.model.model.decode(x)
            write_video(
                os.path.join(cfg.run_dir, "demo1.mp4"),
                y[0] * 0.5 + 0.5,
                fps=8,
            )
        else:
            raise ValueError(f"Unsupported model: {type(trainer.model)}")

        sum_latency: dict[str, float] = {}
        for key, value in results.items():
            module_position_splitted = key.split(".")
            for i in range(len(module_position_splitted)):
                module_position_prefix = ".".join(module_position_splitted[: i + 1])
                if module_position_prefix not in sum_latency:
                    sum_latency[module_position_prefix] = 0.0
                sum_latency[module_position_prefix] += value

        for key, value in sorted(sum_latency.items()):
            print(f"{key} latency: {value:.4f}")

    if cfg.test_decoder_blockwise_latency:
        torch.set_grad_enabled(False)
        x = x.repeat(cfg.batch_size, 1, 1, 1, 1)
        # x = torch.randn(cfg.batch_size, 3, cfg.num_frames, cfg.h, cfg.w, dtype=trainer.amp_dtype, device=device)
        with torch.autocast(device_type="cuda", dtype=trainer.amp_dtype, enabled=True):
            x = trainer.model.encode(x)
        x = x.to(dtype=trainer.amp_dtype, device=device)
        results: dict[str, float] = {}
        if isinstance(trainer.model, DCAEV):
            decoder = trainer.model.decoder.to(dtype=trainer.amp_dtype)
            results["decoder.project_in"] = test_block_latency(
                decoder.project_in, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
            x, _ = decoder.project_in(x)
            for stage_id, stage in reversed(list(enumerate(decoder.stages))):
                if len(stage.op_list) == 0:
                    continue
                depth = decoder.cfg.depth_list[stage_id]
                if stage_id < decoder.num_stages - 1 and depth > 0:
                    results[f"decoder.stage_{stage_id}.upsample"] = test_block_latency(
                        stage.op_list[0], x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                    )
                    x, _ = stage.op_list[0](x)
                    start_block_id = 1
                else:
                    start_block_id = 0
                for block_id in range(depth):
                    block = stage.op_list[start_block_id + block_id]
                    results[f"decoder.stage_{stage_id}.block_{block_id}"] = test_block_latency(
                        block, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
                    )
                    x, _ = block(x)
            results["decoder.project_out"] = test_block_latency(
                decoder.project_out, x, device, trainer.amp_dtype, cfg.warmup_iterations, cfg.iterations
            )
            x, _ = decoder.project_out(x)
        else:
            raise ValueError(f"Unsupported model: {type(trainer.model)}")

        x = x[:, :, trainer.model.cfg.num_pad_frames :]

        write_video(
            os.path.join(cfg.run_dir, "demo_decoder_blockwise_latency.mp4"),
            x[0] * 0.5 + 0.5,
            fps=8,
        )

        sum_latency: dict[str, float] = {}
        for key, value in results.items():
            module_position_splitted = key.split(".")
            for i in range(len(module_position_splitted)):
                module_position_prefix = ".".join(module_position_splitted[: i + 1])
                if module_position_prefix not in sum_latency:
                    sum_latency[module_position_prefix] = 0.0
                sum_latency[module_position_prefix] += value

        for key, value in sorted(sum_latency.items()):
            print(f"{key} latency: {value:.4f}")


if __name__ == "__main__":
    main()
