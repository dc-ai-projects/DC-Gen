import io
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import ipdb
import torch
import torch.nn as nn
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...apps.data_provider.generate_latent_archives import (
    LatentArchivesGenerator,
    LatentArchivesGeneratorConfig,
    SingleVideoTarDataset,
)
from ...apps.data_provider.sampler import AspectRatioBatchSampler, DistributedRangedSampler
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
from ...apps.utils.config import get_config
from ...apps.utils.video import AspectRatioVideoResizeCenterCrop, write_video
from ...models.utils.network import freeze_weights
from ...staecore.autoencoder import Autoencoder, AutoencoderConfig
from ..image_encoder import ImageEncoder, ImageEncoderConfig


@dataclass
class VideoGenCoreLatentArchivesGeneratorConfig(LatentArchivesGeneratorConfig):
    dataset_name: str = "Wan"

    resolution_str: str = MISSING

    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    image_encoder: ImageEncoderConfig = field(
        default_factory=lambda: ImageEncoderConfig(
            resolution="${..resolution_str}", use_vlm="${..get_vlm_feature}", use_mask=False
        )
    )

    batch_sampler: str = "AspectRatioBatchSampler"
    size_transform: str = "AspectRatioVideoResizeCenterCrop"
    multi_scale: bool = False

    video_ext: str = ".mp4"
    num_frames: int = 81

    get_ae_feature: bool = True
    get_vlm_feature: bool = True

    max_batch_num: Optional[int] = None


class VideoGenCoreSingleVideoTarDataset(SingleVideoTarDataset):
    def __init__(self, tar_path: str, video_ext: str, transform: Callable, dataset_name: str, num_frames: int):
        super().__init__(tar_path, video_ext, transform)
        self.dataset_name = dataset_name
        self.transform = transform
        self.num_frames = num_frames

        if dataset_name in ["Wan", "FusionX"]:
            pass
        else:
            raise ValueError(f"dataset {dataset_name} is not supported")

    def get_data_info(self, index: int) -> dict:
        if self.dataset_name in ["Wan", "FusionX"]:
            _, size, offset = self.sample_meta_list[index][".json"]
            stream = io.BytesIO(self._file_data[offset : offset + size])
            data_info = json.load(stream)
        else:
            raise ValueError(f"dataset {self.dataset_name} is not supported")
        return data_info

    def __getitem__(self, index: int):
        sample = super().__getitem__(index)
        captions = []
        clip_scores = []
        corrupted = False

        if sample["raw_num_frames"] < self.num_frames:
            corrupted = True

        if self.dataset_name == "Wan":
            if "captions" in sample[".json"] and "clip_scores" in sample[".json"]:
                captions = sample[".json"]["captions"]
                clip_scores = sample[".json"]["clip_scores"]
            else:
                prompt_key = "extended_prompt" if "extended_prompt" in sample[".json"] else "prompt"
                if prompt_key in sample[".json"]:
                    caption = sample[".json"][prompt_key]
                else:
                    caption = ""
                    corrupted = True
                captions.append(caption)
                clip_scores.append("30.0")
        else:
            raise ValueError(f"dataset {self.dataset_name} is not supported")

        sample[".json"]["corrupted"] = corrupted
        sample[".json"]["captions"] = captions
        sample[".json"]["clip_scores"] = clip_scores
        return sample


class VideoGenCoreLatentArchivesGenerator(LatentArchivesGenerator):
    def __init__(self, cfg: VideoGenCoreLatentArchivesGeneratorConfig):
        if cfg.resolution_str == "480F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F64MS()
        elif cfg.resolution_str == "480F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo480F32MS()
        elif cfg.resolution_str == "720F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo720F64MS()
        elif cfg.resolution_str == "720":
            self.aspect_ratio_manager = AspectRatioManagerVideo720()
        elif cfg.resolution_str == "720F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo720F32MS()
        elif cfg.resolution_str == "1080F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo1080F32MS()
        elif cfg.resolution_str == "1080F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo1080F64MS()
        elif cfg.resolution_str == "2160F32MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo2160F32MS()
        elif cfg.resolution_str == "2160F64MS":
            self.aspect_ratio_manager = AspectRatioManagerVideo2160F64MS()
        else:
            raise ValueError(f"resolution {cfg.resolution_str} is not supported for VideoMSCrop")

        super().__init__(cfg)
        self.cfg: VideoGenCoreLatentArchivesGeneratorConfig
        self.model: Autoencoder

        self.transform = self.build_size_transform()

        self.image_encoder = ImageEncoder(self.cfg.image_encoder, self.device)

    def build_model(self) -> nn.Module:
        model = Autoencoder(self.cfg.autoencoder).to(device=self.device, dtype=self.dtype)
        freeze_weights(model)
        return model

    def build_size_transform(self) -> Callable:
        if self.cfg.size_transform == "AspectRatioVideoResizeCenterCrop":
            size_transform = AspectRatioVideoResizeCenterCrop(self.aspect_ratio_manager, num_frames=self.cfg.num_frames)
        else:
            raise ValueError(f"size transform {self.cfg.size_transform} is not supported")

        return size_transform

    def build_dataset_for_single_archive(self, index: int) -> VideoGenCoreSingleVideoTarDataset:
        dataset = VideoGenCoreSingleVideoTarDataset(
            self.shardlist[index]["url"],
            self.cfg.video_ext,
            self.transform,
            self.cfg.dataset_name,
            self.cfg.num_frames,
        )
        return dataset

    def build_data_loader_for_single_archive(self, index: int) -> DataLoader:
        dataset = self.build_dataset_for_single_archive(index)
        self.sampler = DistributedRangedSampler(dataset, num_replicas=1, rank=0, shuffle=False)
        if self.cfg.batch_sampler == "AspectRatioBatchSampler":
            self.batch_sampler = AspectRatioBatchSampler(
                sampler=self.sampler,
                dataset=dataset,
                batch_size=self.cfg.batch_size,
                save_checkpoint_steps=None,
                aspect_ratio_manager=self.aspect_ratio_manager,
                drop_last=False,
            )
        else:
            raise ValueError(f"batch sampler {self.cfg.batch_sampler} is not supported")
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
        )
        return data_loader

    @torch.no_grad()
    def generate_latent(
        self, video: torch.Tensor, raw_image_list: list[Image.Image]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        video: Tensors of shape (B, C, T, H, W) and range [-1, 1]
        """
        latent = self.model.encode(video)

        if self.cfg.get_ae_feature:
            img_feat = self.image_encoder.get_ae_feature(video[:, :, 0], self.model, num_frames=video.shape[2])
        else:
            img_feat = torch.zeros((5))  # Pad

        if self.cfg.get_vlm_feature:
            clip_feat = torch.cat([self.image_encoder.get_vlm_feature(raw_image) for raw_image in raw_image_list])
        else:
            clip_feat = torch.zeros((5))  # Pad

        return latent, img_feat, clip_feat

    def generate_data_for_single_archive(self, index: int):
        data_loader = self.build_data_loader_for_single_archive(index)

        data_format = {self.cfg.latent_ext: "data", ".json": "data"}
        data_list = []

        idx = 0

        for batch in tqdm(data_loader, desc=f"Generating latent for {self.shardlist[index]['url']}"):
            if self.cfg.max_batch_num is not None and idx >= self.cfg.max_batch_num:
                break

            if batch["video"].shape[2] < self.cfg.num_frames:
                continue  # Too short real data
            raw_image_list, video = batch["raw_image"], batch["video"]
            video = video.to(device=self.device, dtype=self.dtype)
            with torch.no_grad():
                latent, img_feat, clip_feat = self.generate_latent(video, raw_image_list)
            for i in range(len(raw_image_list)):
                sample = {}
                if self.cfg.latent_ext == ".npz":
                    sample[".npz"] = {
                        "latent": latent[i].float().cpu().numpy(),
                        "img_feat": img_feat[i].float().cpu().numpy(),
                        "clip_feat": clip_feat[i].float().cpu().numpy(),
                    }
                elif self.cfg.latent_ext == ".pth":
                    sample[".pth"] = {
                        "latent": latent[i].cpu(),
                        "img_feat": img_feat[i].cpu(),
                        "clip_feat": clip_feat[i].cpu(),
                    }
                else:
                    raise ValueError(f"latent ext {self.cfg.latent_ext} not supported")
                for key, value in batch.items():
                    if key.startswith("."):
                        sample[key] = value[i]
                data_list.append((batch["__key__"][i], sample))

            idx += 1

        new_data_list = []
        for data in data_list:
            if not data[1][".json"]["corrupted"]:
                data[1][".json"].pop("corrupted")
                new_data_list.append(data)

        return data_format, new_data_list


def main():
    cfg: VideoGenCoreLatentArchivesGeneratorConfig = get_config(VideoGenCoreLatentArchivesGeneratorConfig)
    generator = VideoGenCoreLatentArchivesGenerator(cfg)
    generator.generate()


if __name__ == "__main__":
    main()
