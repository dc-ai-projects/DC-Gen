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
import tempfile
from io import BytesIO
from typing import Any, Callable, Optional

import cv2
import imageio
import numpy as np
import pandas
import torch
from PIL import Image
from torch.utils.data.dataset import Subset
from torchvision.datasets.folder import DatasetFolder

from .image import DMCrop, DMCropOrUpsampleCrop, Resize, ResizeCenterCrop, UpsampleCrop


class VideoLoader:
    def __init__(
        self,
        fp: str | BytesIO,
        name: Optional[str] = None,
        crop_range: Optional[tuple[float, float, float, float]] = None,
    ):
        if isinstance(fp, str):
            self.video_capture = cv2.VideoCapture(fp)
            name = name or fp
        elif isinstance(fp, BytesIO):
            # assuming mp4
            with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
                temp_video.write(fp.read())
                temp_video.flush()
                temp_video_name = temp_video.name
                self.video_capture = cv2.VideoCapture(temp_video_name)
                name = name or temp_video_name
        else:
            raise ValueError(f"Type {type(fp)} is not supported")
        self.name = name
        self.crop_range = crop_range

    def get_fps(self):
        return self.video_capture.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count > 0:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if self.video_capture.grab():
                break
            frame_count -= 1
        return frame_count

    def get_contiguous_frames(self, start_frame_index: int, num_frames: int) -> list[Image.Image]:
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
        frames = []
        for _ in range(num_frames):
            success, frame = self.video_capture.read()
            assert success, f"Failed to get contiguous frames from video {self.name}"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.crop_range is not None:
                frame = frame.crop(self.crop_range)
            frames.append(frame)
        return frames

    def get_frames(self, frame_indices: list[int]) -> list[Image.Image]:
        frames = []
        last_frame_index = -100
        for frame_index in frame_indices:
            if frame_index != last_frame_index + 1:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = self.video_capture.read()
            assert success, f"Failed to get frame {frame_index} from video {self.name}"
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            if self.crop_range is not None:
                frame = frame.crop(self.crop_range)
            frames.append(frame)
            last_frame_index = frame_index
        return frames

    def get_all_frames(self) -> list[Image.Image]:
        frame_count = self.get_frame_count()
        return self.get_contiguous_frames(0, frame_count)


def write_video(path: str, images: list[Image.Image] | torch.Tensor, fps: float = 10):
    assert path.endswith(".mp4"), f"only support writing mp4 videos"
    if isinstance(images, list) and all(isinstance(image, Image.Image) for image in images):
        images = [np.array(image) for image in images]
    elif isinstance(images, torch.Tensor):
        # CTHW with value in [0, 1]
        images = (255 * images + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        images = [image for image in images]
    else:
        raise ValueError(f"Type {type(images)} is not supported in write_video")

    video_writer = imageio.get_writer(path, mode="I", fps=fps)
    for image in images:
        video_writer.append_data(image)
    video_writer.close()


class TemporalCenterCrop:
    def __init__(self, t: Optional[int] = None):
        self.t = t

    def __call__(
        self, video: VideoLoader, frame_indices: np.ndarray, t: Optional[int] = None, seed: Optional[int] = None
    ) -> list[Image.Image]:
        if t is None:
            assert self.t is not None
            t = self.t
        center = len(frame_indices) // 2
        frame_indices = frame_indices[
            np.arange(center - t // 2, center + t // 2 + t % 2).clip(0, len(frame_indices) - 1)
        ]
        return video.get_frames(frame_indices)


class TemporalRandomCrop:
    def __init__(self, t: Optional[int] = None):
        self.t = t

    def __call__(
        self, video: VideoLoader, frame_indices: np.ndarray, t: Optional[int] = None, seed: Optional[int] = None
    ) -> list[Image.Image]:
        if t is None:
            assert self.t is not None
            t = self.t

        generator = torch.Generator()
        if seed is not None:
            generator.manual_seed(seed)

        if len(frame_indices) < t:
            frame_indices = np.pad(frame_indices, (0, t - len(frame_indices)), "edge")

        starting_idx = torch.randint(0, len(frame_indices) - t + 1, size=(1,), generator=generator).item()
        frame_indices = frame_indices[np.arange(starting_idx, starting_idx + t)]
        return video.get_frames(frame_indices)


class VideoSizeTransform:
    def __init__(
        self,
        temporal_transform: str,
        spatial_transform: str,
        resample_method: Optional[str] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        t: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        if temporal_transform == "CenterCrop":
            self.temporal_transform = TemporalCenterCrop(t)
        elif temporal_transform == "RandomCrop":
            self.temporal_transform = TemporalRandomCrop(t)
        else:
            raise ValueError(f"temporal transform {temporal_transform} is not supported")

        if spatial_transform == "DMCrop":
            size = None if h is None and w is None else (h, w)
            self.spatial_transform = DMCrop(size)
        elif spatial_transform == "Resize":
            assert (h is None and w is None) or h == w
            self.spatial_transform = Resize(h)
        elif spatial_transform == "UpsampleCrop":
            assert (h is None and w is None) or h == w
            self.spatial_transform = UpsampleCrop(h)
        elif spatial_transform == "DMCropOrUpsampleCrop":
            assert (h is None and w is None) or h == w
            self.spatial_transform = DMCropOrUpsampleCrop(h)
        elif spatial_transform == "ResizeCenterCrop":
            assert (h is None and w is None) or h == w
            self.spatial_transform = ResizeCenterCrop(h)
        else:
            raise ValueError(f"spatial_transform {spatial_transform} is not supported")

        self.resample_method = resample_method
        self.fps = fps

    def __call__(
        self,
        video: VideoLoader,
        original_t: int,
        original_fps: float,
        h: Optional[int] = None,
        w: Optional[int] = None,
        t: Optional[int] = None,
        fps: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        fps = fps or self.fps
        if fps is not None:
            duration = original_t / original_fps
            num_frames_to_sample = int(duration * fps)
            frame_indices = np.linspace(0, original_t - 1, num_frames_to_sample)
            if self.resample_method == "Round":
                frame_indices = frame_indices.round().astype(int)
            elif self.resample_method == "Floor":
                frame_indices = frame_indices.astype(int)
            else:
                raise ValueError(f"resample method {self.resample_method} is not supported")
        else:
            frame_indices = np.arange(original_t)

        frames = self.temporal_transform(video, frame_indices, t, seed)
        if isinstance(self.spatial_transform, DMCrop):
            size = None if h is None and w is None else (h, w)
            frames = [self.spatial_transform(frame, size, seed) for frame in frames]
        elif isinstance(self.spatial_transform, (Resize, UpsampleCrop, DMCropOrUpsampleCrop, ResizeCenterCrop)):
            if h != w:
                raise ValueError(
                    f"the current implementation only supports h ({h}) == w ({w}) for {self.spatial_transform}"
                )
            frames = [self.spatial_transform(frame, h, seed) for frame in frames]
        else:
            raise ValueError(f"spatial_transform {self.spatial_transform} is not supported")
        return frames


def parse_index(index: int | tuple[int, int, int, int, Optional[float], int]) -> tuple:
    if isinstance(index, int):
        h, w, t, fps, seed = None, None, None, None, None
    elif isinstance(index, tuple) and len(index) == 6:
        index, h, w, t, fps, seed = index
    else:
        raise ValueError(f"index {index} is not supported")
    if seed is None:
        seed = int(index)
    return index, h, w, t, fps, seed


class MultiResolutionVideoFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        size_transform: Optional[VideoSizeTransform] = None,
        transform: Optional[Callable] = None,
        return_dict: bool = False,
        metadata: Optional[pandas.DataFrame] = None,
    ) -> None:
        root = os.path.expanduser(root)
        self.root = root
        self.size_transform = size_transform
        self.transform = transform
        self.return_dict = return_dict
        classes, class_to_idx = self.find_classes(root)
        samples = self.make_dataset(root, class_to_idx=class_to_idx, extensions=[".mp4"])
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.metadata = metadata
        if size_transform is not None:
            assert metadata is not None

    def __getitem__(self, index: int | tuple[int, int, Optional[float], int, int]) -> dict[str, Any]:
        index, h, w, t, fps, seed = parse_index(index)

        video_path, target = self.samples[index]
        video = VideoLoader(video_path)
        if self.size_transform is not None:
            video = self.size_transform(
                video, self.metadata.iloc[index]["T"], self.metadata.iloc[index]["fps"], h, w, t, fps, seed
            )
        if self.transform is not None:
            video = torch.stack([self.transform(frame) for frame in video], dim=1)

        if self.return_dict:
            return {
                "index": index,
                "path": video_path,
                "name": os.path.basename(video_path),
                "data": video,
            }
        else:
            return video, target


class MultiResolutionVideoSubset(Subset):
    def __getitem__(
        self, index: int | tuple[int, int, Optional[float], int, int]
    ) -> tuple[torch.Tensor, Any] | dict[str, Any]:
        index, h, w, t, fps, seed = parse_index(index)
        return self.dataset[self.indices[index], h, w, t, fps, seed]

    def __getitems__(self, indices: list[int] | list[tuple[int, int, Optional[float], int, int]]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([(self.indices[index], h, w, t, fps, seed) for index, h, w, t, fps, seed in map(parse_index, indices)])  # type: ignore[attr-defined]
        else:
            return [
                self.dataset[self.indices[index], h, w, t, fps, seed]
                for index, h, w, t, fps, seed in map(parse_index, indices)
            ]
