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

import numbers
import os
import subprocess
import tempfile
from io import BytesIO
from typing import Any, Callable, Optional

import imageio
import ipdb
import numpy as np
import pandas
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets.folder import DatasetFolder

from .aspect_ratio import BaseAspectRatioManager, get_aspect_ratio_manager
from .image import (
    AspectRatioResizeCenterCrop,
    DMCrop,
    DMCropOrUpsampleCrop,
    IdentityTransform,
    Resize,
    ResizeCenterCrop,
    UpsampleCrop,
)


def convert_to_mp4_h264(input_path: str, output_path: str, fps: Optional[float] = None, verbose: bool = False):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-c:v", "libx264"]
    if fps is not None:
        cmd.extend(["-r", str(fps)])
    cmd.append(output_path)
    subprocess.run(
        cmd,
        stderr=subprocess.DEVNULL if not verbose else None,
        check=True,
    )


def crop_video(input_path: str, output_path: str, start_frame: int, end_frame: int, verbose: bool = False):
    """
    crop [start_frame, start_frame + 1, ..., end_frame - 1] from source video
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vf",
            f"select='between(n,{start_frame},{end_frame - 1})'",
            output_path,
        ],
        stderr=subprocess.DEVNULL if not verbose else None,
        check=True,
    )


class VideoLoader:
    def __init__(
        self,
        fp: str | BytesIO,
        name: Optional[str] = None,
        crop_range: Optional[tuple[float, float, float, float]] = None,
    ):
        import cv2

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
        import cv2

        return self.video_capture.get(cv2.CAP_PROP_FPS)

    def get_frame_count(self):
        import cv2

        frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while frame_count > 0:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
            if self.video_capture.grab():
                break
            frame_count -= 1
        return frame_count

    def get_contiguous_frames(self, start_frame_index: int, num_frames: int) -> list[Image.Image]:
        import cv2

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

    def get_frames(self, frame_indices: list[int], frame_access_method: str = "seek") -> list[Image.Image]:
        import cv2

        if frame_access_method.startswith("hybrid_seek_sequential@"):
            interval = int(frame_access_method.removeprefix("hybrid_seek_sequential@"))
        frames = []
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
        last_frame_index = frame_indices[0] - 1
        for frame_index in frame_indices:
            if frame_index == last_frame_index:
                frames.append(frames[-1])
                continue
            if frame_index != last_frame_index + 1:
                if frame_access_method == "seek":
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                elif frame_access_method == "sequential":
                    if frame_index < last_frame_index + 1:
                        raise ValueError(
                            f"frame indices must be increasing for frame_access_method=sequential, but got {last_frame_index} and {frame_index}"
                        )
                    while frame_index > last_frame_index + 1:
                        self.video_capture.read()
                        last_frame_index += 1
                elif frame_access_method.startswith("hybrid_seek_sequential@"):
                    if frame_index < last_frame_index + 1 or frame_index > last_frame_index + interval:
                        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    else:
                        while frame_index > last_frame_index + 1:
                            self.video_capture.read()
                            last_frame_index += 1
                else:
                    raise ValueError(f"frame_access_method {frame_access_method} is not supported")
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


def write_video(path: str | BytesIO, images: list[Image.Image] | torch.Tensor, fps: float = 10):
    if isinstance(path, str):
        assert path.endswith(".mp4"), f"only support writing mp4 videos"
    if isinstance(images, list) and all(isinstance(image, Image.Image) for image in images):
        images = [np.array(image) for image in images]
    elif isinstance(images, torch.Tensor):
        # CTHW with value in [0, 1]
        images = (255 * images + 0.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
        images = [image for image in images]
    else:
        raise ValueError(f"Type {type(images)} is not supported in write_video")

    video_writer = imageio.get_writer(path, mode="I", fps=fps, format="mp4", codec="h264")
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
        return video.get_frames(frame_indices, frame_access_method="sequential")


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
        return video.get_frames(frame_indices, frame_access_method="sequential")


class TemporalFrontCrop:
    def __init__(self, t: Optional[int] = None):
        self.t = t

    def __call__(
        self, video: VideoLoader, frame_indices: np.ndarray, t: Optional[int] = None, seed: Optional[int] = None
    ) -> list[Image.Image]:
        if t is None:
            assert self.t is not None
            t = self.t
        if len(frame_indices) < t:
            frame_indices = np.pad(frame_indices, (0, t - len(frame_indices)), "edge")
        frame_indices = frame_indices[:t]
        return video.get_frames(frame_indices, frame_access_method="sequential")


class TemporalStridedDownsample:
    def __init__(self, stride: int):
        self.stride = stride

    def __call__(
        self, video: VideoLoader, frame_indices: np.ndarray, t: Optional[int] = None, seed: Optional[int] = None
    ) -> list[Image.Image]:
        frame_indices = frame_indices[0 :: self.stride]
        return video.get_frames(frame_indices, frame_access_method="sequential")


class TemporalKeepAll:
    def __init__(self):
        pass

    def __call__(
        self, video: VideoLoader, frame_indices: np.ndarray, t: Optional[int] = None, seed: Optional[int] = None
    ) -> list[Image.Image]:
        return video.get_frames(frame_indices, frame_access_method="sequential")


class VideoSizeTransform:
    def __init__(
        self,
        temporal_transform: str,
        spatial_transform: Optional[str],
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
        elif temporal_transform == "FrontCrop":
            self.temporal_transform = TemporalFrontCrop(t)
        elif temporal_transform.startswith("StridedDownsample@"):
            self.temporal_transform = TemporalStridedDownsample(int(temporal_transform.split("@")[1]))
        elif temporal_transform == "KeepAll":
            self.temporal_transform = TemporalKeepAll()
        else:
            raise ValueError(f"temporal transform {temporal_transform} is not supported")

        if spatial_transform is None:
            self.spatial_transform = None
        elif spatial_transform == "DMCrop":
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
        elif spatial_transform.startswith("AspectRatioResizeCenterCrop@"):
            resolution = spatial_transform.removeprefix("AspectRatioResizeCenterCrop@")
            aspect_ratio_manager = get_aspect_ratio_manager(resolution)
            self.spatial_transform = AspectRatioResizeCenterCrop(aspect_ratio_manager)
        else:
            raise ValueError(f"spatial_transform {spatial_transform} is not supported")

        self.resample_method = resample_method
        self.fps = fps

    def __call__(
        self,
        video: VideoLoader,
        original_t: Optional[int] = None,
        original_fps: Optional[float] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        t: Optional[int] = None,
        fps: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> list[Image.Image]:
        fps = fps or self.fps
        original_t = original_t or video.get_frame_count()
        if fps is not None:
            original_fps = original_fps or video.get_fps()
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
        if self.spatial_transform is None:
            pass
        elif isinstance(self.spatial_transform, DMCrop):
            size = None if h is None and w is None else (h, w)
            frames = [self.spatial_transform(frame, size, seed) for frame in frames]
        elif isinstance(self.spatial_transform, (Resize, UpsampleCrop, DMCropOrUpsampleCrop, ResizeCenterCrop)):
            if h != w:
                raise ValueError(
                    f"the current implementation only supports h ({h}) == w ({w}) for {self.spatial_transform}"
                )
            frames = [self.spatial_transform(frame, h, seed) for frame in frames]
        elif isinstance(self.spatial_transform, AspectRatioResizeCenterCrop):
            frames = [self.spatial_transform(frame) for frame in frames]
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
                "video": video,
            }
        else:
            return video, target


class MultiResolutionVideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        size_transform: Optional[VideoSizeTransform] = None,
        transform: Optional[Callable] = None,
        metadata: Optional[pandas.DataFrame] = None,
    ) -> None:
        root = os.path.expanduser(root)
        self.root = root
        self.size_transform = size_transform
        self.transform = transform
        samples = []
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".mp4"):
                    samples.append(os.path.join(dirpath, filename))
        self.samples = sorted(samples)
        self.metadata = metadata
        if size_transform is not None and not isinstance(size_transform, IdentityTransform):
            assert metadata is not None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int | tuple[int, int, Optional[float], int, int]) -> dict[str, Any]:
        index, h, w, t, fps, seed = parse_index(index)

        video_path = self.samples[index]
        video = VideoLoader(video_path)
        if self.size_transform is not None and not isinstance(self.size_transform, IdentityTransform):
            video = self.size_transform(
                video, self.metadata.at[index, "T"], self.metadata.at[index, "fps"], h, w, t, fps, seed
            )  # self.metadata.iloc[index]["T"] may change data type
        if self.transform is not None and not isinstance(self.transform, IdentityTransform):
            video = torch.stack([self.transform(frame) for frame in video], dim=1)

        return {"video": video}


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


class AspectRatioVideoResizeCenterCrop:
    """multi-aspect-ratio resize and center crop"""

    # https://github.com/Efficient-Large-Model/Sana/blob/video/Sana-video/diffusion/data/video_transforms.py

    def __init__(self, aspect_ratio_manager: BaseAspectRatioManager, num_frames: int = 81) -> None:
        self.aspect_ratio_manager = aspect_ratio_manager
        self.num_frames = num_frames
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)

    def resize(self, clip, target_size, interpolation_mode):
        if len(target_size) != 2:
            raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
        return torch.nn.functional.interpolate(clip, size=target_size, mode=interpolation_mode, align_corners=False)

    def crop(self, clip, i, j, h, w):
        if len(clip.size()) != 4:
            raise ValueError("clip should be a 4D tensor")
        return clip[..., i : i + h, j : j + w]

    def resize_crop_to_fill(self, clip, target_size):
        h, w = clip.size(-2), clip.size(-1)
        th, tw = target_size[0], target_size[1]
        rh, rw = th / h, tw / w
        if rh > rw:
            sh, sw = th, round(w * rh)
            clip = self.resize(clip, (sh, sw), "bilinear")
            i = 0
            j = int(round(sw - tw) / 2.0)
        else:
            sh, sw = round(h * rw), tw
            clip = self.resize(clip, (sh, sw), "bilinear")
            i = int(round(sh - th) / 2.0)
            j = 0
        assert i + th <= clip.size(-2) and j + tw <= clip.size(-1)
        return self.crop(clip, i, j, th, tw)

    def __call__(self, video: VideoLoader) -> torch.Tensor:
        video_num_frames = video.get_frame_count()
        if video_num_frames < self.num_frames:
            print(f"warning: num frames {video.get_frame_count()} smaller than required num frames {self.num_frames}")
            frames = video.get_contiguous_frames(0, video_num_frames)
            frames = frames + [frames[-1]] * (self.num_frames - video_num_frames)
        else:
            frames = video.get_contiguous_frames(0, self.num_frames)
        clip = torch.stack([torch.from_numpy(np.array(frame)) for frame in frames], dim=0).permute(0, 3, 1, 2)

        _, _, height, width = clip.shape
        closest_ratio = self.aspect_ratio_manager.get_closest_ratio(height, width)
        closest_size = self.aspect_ratio_manager.get_dimensions(closest_ratio)

        assert clip.dtype == torch.uint8, "clip tensor should have data type uint8. Got %s" % str(clip.dtype)
        clip = clip.float() / 255.0  # TCHW
        clip = self.resize_crop_to_fill(clip, closest_size)
        clip = self.normalize(clip)
        clip = clip.permute(1, 0, 2, 3)
        return clip
