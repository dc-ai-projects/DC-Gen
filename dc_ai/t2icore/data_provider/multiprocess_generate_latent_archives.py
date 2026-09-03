# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate image latent archives with persistent multiprocess workers."""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import ray
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from ...aecore.autoencoder import Autoencoder, AutoencoderConfig
from ...apps.data_provider.generate_latent_archives import SingleImageTarDataset
from ...apps.data_provider.sampler import AspectRatioBatchSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManager512,
    AspectRatioManager1024,
    AspectRatioManager2048,
    AspectRatioManager4096,
    BaseAspectRatioManager,
)
from ...apps.utils.config import get_config
from ...apps.utils.dtype import get_dtype_from_str
from ...apps.utils.io import read_json
from ...apps.utils.multiprocess import Distributor, MultiProcessConfig, SubtaskStatus, Worker
from ...apps.utils.tar import TarWriter
from .image import (
    build_image_transform,
    get_image_captions,
    get_image_dimensions,
    get_image_sidecar_entry,
)

_SUPPORTED_RESOLUTIONS = (512, 1024, 2048, 4096)
RESOURCE_EPSILON = 1e-9


@dataclass
class MultiprocessGenerateLatentArchivesConfig(MultiProcessConfig):
    use_ray: bool = False
    num_workers: int = 1

    meta_path: str = MISSING
    original_archive_dir: str = MISSING
    save_dir: str = MISSING
    num_subtasks_per_archive: Optional[int] = None

    resolution: int = MISSING
    batch_size: int = MISSING
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)

    image_ext: str = ".jpg"
    mean: float = 0.5
    std: float = 0.5
    dtype: str = "fp32"
    latent_dtype: str = "fp32"
    num_dataloader_workers: int = 0

    num_cpus_per_worker: int = 4
    num_gpus_per_worker: float = 1.0

    seed: int = 0
    debug: bool = False


@dataclass(frozen=True)
class ImageLatentSubtask:
    subtask_id: int
    input_path: str
    save_path: str
    split_index: Optional[int] = None


def _validate_config(cfg: MultiprocessGenerateLatentArchivesConfig) -> None:
    if cfg.resolution not in _SUPPORTED_RESOLUTIONS:
        raise ValueError(f"resolution {cfg.resolution} is not supported; choose from {_SUPPORTED_RESOLUTIONS}")
    if cfg.batch_size <= 0 or cfg.num_workers <= 0:
        raise ValueError("batch_size and num_workers must be positive")
    if not cfg.use_ray and cfg.num_workers != 1:
        raise ValueError("num_workers must be 1 when use_ray=False")
    if cfg.num_dataloader_workers < 0:
        raise ValueError("num_dataloader_workers must be non-negative")
    if cfg.num_cpus_per_worker <= 0:
        raise ValueError("num_cpus_per_worker must be positive")
    if cfg.num_gpus_per_worker < 0:
        raise ValueError("num_gpus_per_worker must be non-negative")
    if cfg.num_subtasks_per_archive is not None and cfg.num_subtasks_per_archive <= 0:
        raise ValueError("num_subtasks_per_archive must be positive when provided")
    if cfg.num_subtasks_per_task is not None and cfg.num_subtasks_per_task <= 0:
        raise ValueError("num_subtasks_per_task must be positive when provided")


def _aspect_ratio_manager(resolution: int) -> BaseAspectRatioManager:
    if resolution == 512:
        return AspectRatioManager512()
    if resolution == 1024:
        return AspectRatioManager1024()
    if resolution == 2048:
        return AspectRatioManager2048()
    if resolution == 4096:
        return AspectRatioManager4096()
    raise ValueError(f"resolution {resolution} is not supported; choose from {_SUPPORTED_RESOLUTIONS}")


def get_required_execution_resources(cfg: MultiprocessGenerateLatentArchivesConfig) -> tuple[int, float]:
    if cfg.use_ray:
        distributor_cpus = 1
        required_cpus = cfg.num_workers * cfg.num_cpus_per_worker + distributor_cpus
        required_gpus = cfg.num_workers * cfg.num_gpus_per_worker
        return required_cpus, required_gpus
    return cfg.num_cpus_per_worker, cfg.num_gpus_per_worker


def _validate_ray_resources(cfg: MultiprocessGenerateLatentArchivesConfig, resources: dict[str, float]) -> None:
    required_cpus, required_gpus = get_required_execution_resources(cfg)
    available_cpus = resources.get("CPU", 0)
    available_gpus = resources.get("GPU", 0)
    if available_cpus < required_cpus or available_gpus + RESOURCE_EPSILON < required_gpus:
        raise RuntimeError(
            "Ray must be able to schedule the distributor and all workers simultaneously; "
            f"requires CPU>={required_cpus} and GPU>={required_gpus}, "
            f"but found CPU={available_cpus} and GPU={available_gpus}"
        )


class SidecarImageTarDataset(SingleImageTarDataset):
    """Read images from a tar archive and normalized metadata from its sidecar JSON."""

    def __init__(self, tar_path: str, image_ext: str, transform: Optional[Callable]) -> None:
        super().__init__(tar_path, image_ext, transform)
        self.sidecar_path = tar_path.removesuffix(".tar") + ".json"
        if not os.path.isfile(self.sidecar_path):
            raise FileNotFoundError(
                f"Image metadata sidecar does not exist: {self.sidecar_path}; "
                "generate KreaGen sidecars with "
                "python -m dc_ai.t2icore.data_provider.extract_krea_gen_sidecars"
            )
        self.sidecar = read_json(self.sidecar_path)
        if not isinstance(self.sidecar, dict):
            raise ValueError(f"Image metadata sidecar must contain an object: {self.sidecar_path}")

    def _get_entry(self, index: int) -> dict:
        key = self.sample_prefix_list[index]
        return get_image_sidecar_entry(self.sidecar, self.sidecar_path, key)

    def _get_dimensions(self, index: int) -> Optional[dict[str, int]]:
        key = self.sample_prefix_list[index]
        entry = self._get_entry(index)
        return get_image_dimensions(entry, self.sidecar_path, key)

    def _get_captions(self, index: int) -> tuple[list[str], list[object]]:
        key = self.sample_prefix_list[index]
        entry = self._get_entry(index)
        return get_image_captions(entry, self.sidecar_path, key)

    def get_data_info(self, index: int) -> Optional[dict[str, int]]:
        return self._get_dimensions(index)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        dimensions = self._get_dimensions(index)
        if dimensions is None:
            raise RuntimeError(f"Corrupted sample {sample['__key__']!r} must not be loaded")
        captions, clip_scores = self._get_captions(index)
        sample[".json"] = {
            **dimensions,
            "captions": captions,
            "clip_scores": clip_scores,
            "corrupted": False,
        }
        return sample


class MultiprocessGenerateLatentArchivesDistributor(Distributor):
    def __init__(self, cfg: MultiprocessGenerateLatentArchivesConfig):
        _validate_config(cfg)
        super().__init__(cfg)
        self.cfg: MultiprocessGenerateLatentArchivesConfig

    def build_all_subtasks(self) -> dict[int, ImageLatentSubtask]:
        with open(self.cfg.meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if "shardlist" not in metadata or not isinstance(metadata["shardlist"], list):
            raise ValueError(f"WIDS metadata {self.cfg.meta_path} must contain a shardlist")

        original_archive_dir = os.path.abspath(os.path.expanduser(self.cfg.original_archive_dir))
        subtask_list: list[ImageLatentSubtask] = []
        for shard in metadata["shardlist"]:
            if "url" not in shard:
                raise ValueError(f"WIDS shard in {self.cfg.meta_path} is missing url: {shard}")
            input_path = os.path.expanduser(shard["url"])
            if metadata.get("path_format", "absolute") == "relative":
                input_path = os.path.abspath(os.path.join(self.cfg.meta_path, input_path))
            else:
                input_path = os.path.abspath(input_path)
            if os.path.commonpath((original_archive_dir, input_path)) != original_archive_dir:
                raise ValueError(f"Input archive {input_path} is outside original_archive_dir {original_archive_dir}")

            relative_archive_path = os.path.relpath(input_path, original_archive_dir)
            save_path = os.path.splitext(os.path.join(self.cfg.save_dir, relative_archive_path))[0] + ".tar"
            if self.cfg.num_subtasks_per_archive is None:
                subtask_list.append(
                    ImageLatentSubtask(
                        subtask_id=len(subtask_list),
                        input_path=input_path,
                        save_path=save_path,
                    )
                )
            else:
                for split_index in range(self.cfg.num_subtasks_per_archive):
                    subtask_list.append(
                        ImageLatentSubtask(
                            subtask_id=len(subtask_list),
                            input_path=input_path,
                            save_path=save_path.removesuffix(".tar") + f"_{split_index}.tar",
                            split_index=split_index,
                        )
                    )
        return {subtask.subtask_id: subtask for subtask in subtask_list}


class MultiprocessGenerateLatentArchivesWorker(Worker):
    def __init__(
        self,
        cfg: MultiprocessGenerateLatentArchivesConfig,
        distributor: MultiprocessGenerateLatentArchivesDistributor,
        worker_id: int,
    ) -> None:
        _validate_config(cfg)
        super().__init__(cfg, distributor, worker_id)
        self.cfg: MultiprocessGenerateLatentArchivesConfig
        self.setup_env()
        self.setup_seed()
        self.dtype = get_dtype_from_str(cfg.dtype)
        self.latent_dtype = get_dtype_from_str(cfg.latent_dtype)
        self.aspect_ratio_manager = _aspect_ratio_manager(cfg.resolution)
        self.transform = build_image_transform(self.aspect_ratio_manager, cfg.mean, cfg.std)
        self.autoencoder = Autoencoder(cfg.autoencoder).to(device=self.device, dtype=self.dtype)

    def setup_env(self) -> None:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def setup_seed(self) -> None:
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        collated_batch = {}
        for key in batch[0]:
            if key == "image":
                collated_batch[key] = default_collate([item[key] for item in batch])
            else:
                collated_batch[key] = [item[key] for item in batch]
        return collated_batch

    def _build_data_loader(self, input_path: str) -> DataLoader:
        dataset = SidecarImageTarDataset(
            input_path,
            self.cfg.image_ext,
            self.transform,
        )
        sampler = SequentialSampler(dataset)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=None,
            aspect_ratio_manager=self.aspect_ratio_manager,
            drop_last=False,
        )
        return DataLoader(
            dataset=dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.num_dataloader_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=self.collate_fn,
        )

    @torch.no_grad()
    def do_subtask(self, subtask: ImageLatentSubtask) -> SubtaskStatus:
        os.makedirs(os.path.dirname(subtask.save_path), exist_ok=True)
        data_loader = self._build_data_loader(subtask.input_path)
        start_batch_idx, end_batch_idx = None, None
        if subtask.split_index is not None:
            assert self.cfg.num_subtasks_per_archive is not None
            start_batch_idx = subtask.split_index * len(data_loader) // self.cfg.num_subtasks_per_archive
            if subtask.split_index < self.cfg.num_subtasks_per_archive - 1:
                end_batch_idx = (subtask.split_index + 1) * len(data_loader) // self.cfg.num_subtasks_per_archive

        tar_writer = TarWriter(subtask.save_path, {".pth": "data", ".json": "data"})
        latent_total_count = 0
        latent_total_sum_squared = 0.0
        try:
            for batch_idx, batch in enumerate(tqdm(data_loader, desc=f"Generating latent for {subtask.input_path}")):
                if start_batch_idx is not None and batch_idx < start_batch_idx:
                    continue
                if end_batch_idx is not None and batch_idx >= end_batch_idx:
                    break

                images = batch["image"].to(device=self.device, dtype=self.dtype)
                latents = self.autoencoder.encode(images).to(device="cpu", dtype=self.latent_dtype)
                for sample_idx, (key, metadata) in enumerate(zip(batch["__key__"], batch[".json"])):
                    if metadata["corrupted"]:
                        continue
                    metadata = dict(metadata)
                    metadata.pop("corrupted")
                    tar_writer.add_data(key, {".json": metadata, ".pth": latents[sample_idx]})

                latent_total_count += latents.numel()
                latent_total_sum_squared += latents.float().square().sum().item()
                if self.cfg.debug and (
                    (start_batch_idx is None and batch_idx == 4)
                    or (start_batch_idx is not None and batch_idx == start_batch_idx + 4)
                ):
                    break
        finally:
            tar_writer.close()

        if latent_total_count > 0:
            print(f"rms: {np.sqrt(latent_total_sum_squared / latent_total_count)}")
        else:
            print("rms: no latent values generated")
        return SubtaskStatus.COMPLETED


def main() -> None:
    cfg = get_config(MultiprocessGenerateLatentArchivesConfig)
    _validate_config(cfg)
    if cfg.use_ray:
        ray.init(ignore_reinit_error=True, _temp_dir=os.path.expanduser("~/.cache/ray"))
        _validate_ray_resources(cfg, ray.cluster_resources())

        @ray.remote(num_cpus=1)
        class MultiprocessGenerateLatentArchivesDistributorRemote(MultiprocessGenerateLatentArchivesDistributor):
            pass

        @ray.remote(num_cpus=cfg.num_cpus_per_worker, num_gpus=cfg.num_gpus_per_worker)
        class MultiprocessGenerateLatentArchivesWorkerRemote(MultiprocessGenerateLatentArchivesWorker):
            pass

        distributor = MultiprocessGenerateLatentArchivesDistributorRemote.remote(cfg)
        workers = [
            MultiprocessGenerateLatentArchivesWorkerRemote.remote(cfg, distributor, worker_id)
            for worker_id in range(cfg.num_workers)
        ]
        ray.get([worker.work.remote() for worker in workers])
    else:
        distributor = MultiprocessGenerateLatentArchivesDistributor(cfg)
        worker = MultiprocessGenerateLatentArchivesWorker(cfg, distributor, 0)
        worker.work()


if __name__ == "__main__":
    main()
