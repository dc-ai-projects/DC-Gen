from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import MISSING
from torch.utils.data import DataLoader, Dataset, default_collate

from ...apps.data_provider.dc_mixture import (
    MixtureDataProvider,
    MixtureDataProviderConfig,
    MixtureDataset,
    MixtureSampler,
)
from ...apps.data_provider.sampler import DistributedRangedSampler, MixtureAspectRatioBatchSampler
from ...apps.utils.aspect_ratio import (
    AspectRatioManagerVideo480F32MS,
    AspectRatioManagerVideo480F64MS,
    AspectRatioManagerVideo720,
    AspectRatioManagerVideo720F32MS,
    AspectRatioManagerVideo720F64MS,
    AspectRatioManagerVideo1080F32MS,
    AspectRatioManagerVideo1080F64MS,
    AspectRatioManagerVideo2160F32MS,
    AspectRatioManagerVideo2160F64MS,
)
from .collection import possible_train_data_providers


@dataclass
class VideoGenCoreLatentMixtureDataProviderConfig(MixtureDataProviderConfig):
    name: str = "VideoGenCoreLatentMixture"
    data_providers: tuple[str, ...] = ("LatentFusionX",)
    cache_train_states: bool = False
    resolution: str = "480F32MS"  # resolution for training
    shuffle_chunk_size: Optional[int] = 1000
    wds_meta_dir: str = "assets/data/meta"
    multi_scale: bool = False
    vlm_name: Optional[str] = None
    latent_ext: str = ".pth"


class VideoGenCoreLatentMixtureDataset(MixtureDataset):
    def __getitem__(self, index: dict[str, Any]) -> dict[str, Any]:
        sample = self.datasets[index["dataset_index"]][index["sample_index"], index["seed"]]  # No need for resolution
        sample.update({"dataset_name": self.cfg.data_providers[index["dataset_index"]], "index": index})
        return sample

    def get_data_info(self, index: dict[str, Any]):
        sample = self.__getitem__(index)
        return {
            "height": sample["height"],
            "width": sample["width"],
        }


class VideoGenCoreLatentMixtureSampler(MixtureSampler):
    def __init__(
        self,
        cfg: VideoGenCoreLatentMixtureDataProviderConfig,
        datasets: list[Dataset],
        samplers: list[DistributedRangedSampler],
    ):
        super().__init__(cfg, datasets, samplers)
        self.cfg: VideoGenCoreLatentMixtureDataProviderConfig


class VideoGenCoreLatentMixtureDataProvider(MixtureDataProvider):
    def __init__(self, cfg: VideoGenCoreLatentMixtureDataProviderConfig):
        if cfg.resolution == "480F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F64MS()
        elif cfg.resolution == "480F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F32MS()
        elif cfg.resolution == "720F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo720F64MS()
        elif cfg.resolution == "720":
            self.aspect_ratio_manager = AspectRatioManagerVideo720()
        elif cfg.resolution == "720F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo720F32MS()
        elif cfg.resolution == "1080F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo1080F32MS()
        elif cfg.resolution == "1080F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo1080F64MS()
        elif cfg.resolution == "2160F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo2160F32MS()
        elif cfg.resolution == "2160F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo2160F64MS()
        else:
            raise ValueError(f"resolution {cfg.resolution} is not supported")

        super().__init__(cfg)
        self.cfg: VideoGenCoreLatentMixtureDataProviderConfig
        self.sampler: MixtureAspectRatioBatchSampler

    def build_datasets_and_samplers(self) -> tuple[list[Dataset], list[DistributedRangedSampler]]:
        generator = torch.Generator(device=torch.device("cpu"))
        generator.manual_seed(self.cfg.seed)
        datasets: list[Dataset] = []
        samplers: list[DistributedRangedSampler] = []
        for data_provider_name in self.cfg.data_providers:
            if data_provider_name in possible_train_data_providers:
                seed = torch.randint(0, 2**63 - 1, (1,), generator=generator).item()
                data_provider_cfg = possible_train_data_providers[data_provider_name][0](
                    resolution=self.cfg.resolution,
                    wds_meta_dir=self.cfg.wds_meta_dir,
                    seed=seed,
                    shuffle_chunk_size=self.cfg.shuffle_chunk_size,
                    vlm_name=self.cfg.vlm_name,
                    latent_ext=self.cfg.latent_ext,
                )
                data_provider = possible_train_data_providers[data_provider_name][1](data_provider_cfg)
            else:
                raise ValueError(f"data provider {data_provider_name} is not supported in mixture data provider")
            datasets.append(data_provider.dataset)
            samplers.append(data_provider.sampler.sampler)
        return datasets, samplers

    def build_complete_dataset(self) -> VideoGenCoreLatentMixtureDataset:
        return VideoGenCoreLatentMixtureDataset(self.cfg, self.datasets)

    def build_filtered_dataset(self, complete_dataset: Dataset, mask: bool | np.ndarray) -> Dataset:
        if mask == True:
            return complete_dataset
        else:
            raise ValueError(f"mask {mask} is not supported for T2ICoreLatentMixtureDataProvider")

    def build_sampler(self) -> MixtureAspectRatioBatchSampler:
        raw_sampler = VideoGenCoreLatentMixtureSampler(self.cfg, self.datasets, self.samplers)
        sampler = MixtureAspectRatioBatchSampler(
            sampler=raw_sampler,
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            save_checkpoint_steps=self.cfg.save_checkpoint_steps,
            aspect_ratio_manager=self.aspect_ratio_manager,
            drop_last=False,
        )
        return sampler

    def build_data_loader(self) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)
        data_loader = DataLoader(
            dataset=self.dataset,
            batch_sampler=self.sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            generator=generator,
            prefetch_factor=self.cfg.prefetch_factor,
            persistent_workers=self.cfg.persistent_workers,
        )
        return data_loader

    def collate_fn(self, batch):
        videos = default_collate([item["videos"] for item in batch])
        ae_feature = default_collate([item["ae_feature"] for item in batch])
        vlm_feature = (
            default_collate([item["vlm_feature"] for item in batch]) if batch[0]["vlm_feature"] is not None else None
        )
        batch = {
            key: [item[key] for item in batch] for key in batch[0] if key not in ["videos", "img_feat", "clip_feat"]
        }
        batch["videos"], batch["ae_feature"], batch["vlm_feature"] = videos, ae_feature, vlm_feature
        return batch
