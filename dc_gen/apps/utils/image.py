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
import pathlib
from typing import Any, Callable, Optional

import numpy as np
import pandas
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import functional as torchvision_F

from .aspect_ratio import BaseAspectRatioManager

__all__ = ["load_image", "load_image_from_dir", "DMCrop", "CustomImageFolder", "ImageDataset"]


def load_image(data_path: str, mode="rgb") -> Image.Image:
    img = Image.open(data_path)
    if mode == "rgb":
        img = img.convert("RGB")
    return img


def load_image_from_dir(
    dir_path: str,
    suffix: str | tuple[str, ...] | list[str] = (".jpg", ".JPEG", ".png"),
    return_mode="path",
    k: Optional[int] = None,
    shuffle_func: Optional[Callable] = None,
) -> list | tuple[list, list]:
    suffix = [suffix] if isinstance(suffix, str) else suffix

    file_list = []
    for dirpath, _, fnames in os.walk(dir_path):
        for fname in fnames:
            if pathlib.Path(fname).suffix not in suffix:
                continue
            image_path = os.path.join(dirpath, fname)
            file_list.append(image_path)

    if shuffle_func is not None and k is not None:
        shuffle_file_list = shuffle_func(file_list)
        file_list = shuffle_file_list or file_list
        file_list = file_list[:k]

    file_list = sorted(file_list)

    if return_mode == "path":
        return file_list
    else:
        files = []
        path_list = []
        for file_path in file_list:
            try:
                files.append(load_image(file_path))
                path_list.append(file_path)
            except Exception:
                print(f"Fail to load {file_path}")
        if return_mode == "image":
            return files
        else:
            return path_list, files


def convert_image_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    elif image.mode in ["P", "RGBA", "LA"]:
        image_rgba: Image.Image = image.convert("RGBA") if image.mode != "RGBA" else image
        # reference: https://stackoverflow.com/questions/9166400/convert-rgba-png-to-rgb-with-pil
        image_rgba.load()  # required for png.split()
        image = Image.new("RGB", image_rgba.size, (255, 255, 255))
        image.paste(image_rgba, mask=image_rgba.split()[3])
        return image
    elif image.mode in [
        "1",
        "L",
        "CMYK",
        "RGBX",
        "I",
        "I;16",
    ]:  # Pillow can behave differently on different machines. It is possible that on one machine, the mode is "I", while on another it is "I;16".
        return image.convert("RGB")
    else:
        raise ValueError(f"image mode {image.mode} is not supported")


class IdentityTransform:
    def __call__(self, x, *args, **kwargs):
        return x


class Resize:
    def __init__(self, size: Optional[int] = None, method: str = "bicubic") -> None:
        self.size = size
        if method == "bicubic":
            self.method = transforms.InterpolationMode.BICUBIC
        elif method == "bilinear":
            self.method = transforms.InterpolationMode.BILINEAR
        else:
            raise ValueError(f"method {method} is not supported for resizing")

    def __call__(
        self, pil_image: Image.Image, image_size: Optional[int] = None, seed: Optional[int] = None
    ) -> Image.Image:
        if image_size is None:
            assert self.size is not None
            image_size = self.size
        if pil_image.size == (image_size, image_size):
            return pil_image

        pil_image = torchvision_F.resize(pil_image, (image_size, image_size), self.method, None, True)
        return pil_image


class ResizeCenterCrop:
    def __init__(self, size: Optional[int] = None, method: str = "bilinear") -> None:
        self.size = size
        if method == "bilinear":
            self.method = transforms.InterpolationMode.BILINEAR
        else:
            raise ValueError(f"method {method} is not supported for resizing")

    def __call__(
        self, pil_image: Image.Image, image_size: Optional[int] = None, seed: Optional[int] = None
    ) -> Image.Image:
        if image_size is None:
            assert self.size is not None
            image_size = self.size
        if pil_image.size == (image_size, image_size):
            return pil_image

        pil_image = torchvision_F.resize(pil_image, image_size, self.method, None, True)
        pil_image = torchvision_F.center_crop(pil_image, (image_size, image_size))
        return pil_image


class DMCrop:
    """center/random crop used in diffusion models"""

    def __init__(self, size: Optional[int | tuple[int, int]] = None) -> None:
        self.size = size

    def __call__(
        self, pil_image: Image.Image, image_size: Optional[int | tuple[int, int]] = None, seed: Optional[int] = None
    ) -> Image.Image:
        """
        Center cropping implementation from ADM.
        https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

        pil image size: (w, h)
        image_size: (h, w)
        """
        if image_size is None:
            assert self.size is not None
            image_size = self.size

        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if pil_image.size[0] == image_size[1] and pil_image.size[1] == image_size[0]:
            return pil_image

        while pil_image.size[0] >= 2 * image_size[1] and pil_image.size[1] >= 2 * image_size[0]:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

        scale = max(image_size[0] / pil_image.size[1], image_size[1] / pil_image.size[0])
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)
        crop_h = (pil_image.size[1] - image_size[0]) // 2
        crop_w = (pil_image.size[0] - image_size[1]) // 2
        pil_image = pil_image.crop((crop_w, crop_h, crop_w + image_size[1], crop_h + image_size[0]))

        return pil_image


class UpsampleCrop:
    """first do upsample so that the short side >= size, then random crop"""

    def __init__(self, size: Optional[int] = None) -> None:
        self.size = size

    def __call__(self, image: Image.Image, image_size: Optional[int] = None, seed: Optional[int] = None) -> Image.Image:
        # first do upsample so that the short side >= size
        if image_size is None:
            assert self.size is not None
            image_size = self.size
        if min(*image.size) < image_size:
            assert False
            scale = image_size / min(*image.size)
            image = image.resize(tuple(round(x * scale) for x in image.size), resample=Image.BICUBIC)

        # random crop
        w, h = image.size[0], image.size[1]
        generator = torch.Generator()
        generator.manual_seed(seed)
        i = torch.randint(0, w - image_size + 1, size=(1,), generator=generator).item()
        j = torch.randint(0, h - image_size + 1, size=(1,), generator=generator).item()
        image = image.crop((i, j, i + image_size, j + image_size))

        return image


class DMCropOrUpsampleCrop:
    def __init__(self, size: Optional[int] = None) -> None:
        self.dm_crop = DMCrop(size)
        self.upsample_crop = UpsampleCrop(size)

    def __call__(self, image: Image.Image, image_size: Optional[int] = None, seed: Optional[int] = None) -> Image.Image:
        generator = torch.Generator()
        generator.manual_seed(seed)
        if torch.randn((1,), generator=generator).item() < 0.5:
            image = self.dm_crop(image, image_size, seed)
        else:
            image = self.upsample_crop(image, image_size, seed)
        return image


class AspectRatioResizeCenterCrop:
    """multi-aspect-ratio resize and center crop used in t2icore"""

    def __init__(self, aspect_ratio_manager: BaseAspectRatioManager) -> None:
        self.aspect_ratio_manager = aspect_ratio_manager

    def __call__(self, pil_image: Image.Image) -> Image.Image:
        width, height = pil_image.size
        closest_ratio = self.aspect_ratio_manager.get_closest_ratio(height, width)
        closest_size = self.aspect_ratio_manager.get_dimensions(closest_ratio)

        # note: this is different from scale = max(image_size[0] / pil_image.size[1], image_size[1] / pil_image.size[0]), due to rounding error. int(639.99)=639
        if closest_size[0] / height > closest_size[1] / width:
            resize_size = closest_size[0], int(width * closest_size[0] / height)
        else:
            resize_size = int(height * closest_size[1] / width), closest_size[1]

        pil_image = torchvision_F.resize(pil_image, resize_size, transforms.InterpolationMode.BICUBIC)
        pil_image = torchvision_F.center_crop(pil_image, closest_size)

        return pil_image


class CustomImageFolder(ImageFolder):
    def __init__(self, root: str, transform: Optional[Callable] = None, return_dict: bool = False):
        root = os.path.expanduser(root)
        self.return_dict = return_dict
        super().__init__(root, transform)

    def __getitem__(self, index: int) -> dict[str, Any] | tuple[Any, Any]:
        path, target = self.samples[index]
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_dict:
            return {
                "index": index,
                "image_path": path,
                "image": image,
                "label": target,
            }
        else:
            return image, target


def parse_index(index: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int]:
    if isinstance(index, int):
        resolution, seed = None, None
    elif isinstance(index, tuple) and len(index) == 2:
        index, resolution = index
        seed = None
    elif isinstance(index, tuple) and len(index) == 3:
        index, resolution, seed = index
    else:
        raise ValueError(f"index {index} is not supported")
    return index, resolution, seed


class MultiResolutionImageFolder(ImageFolder):
    def __init__(
        self, root: str, size_transform: Callable, transform: Callable, vlm_caption: Optional[pandas.DataFrame] = None
    ):
        super().__init__(root, transform)
        self.size_transform = size_transform
        self.vlm_caption = vlm_caption

    def __getitem__(self, index: int | tuple[int, int] | tuple[int, int, int]) -> tuple[torch.Tensor, Any]:
        index, resolution, seed = parse_index(index)
        path, label = self.samples[index]
        image = load_image(path)
        image = self.size_transform(image, resolution, seed)
        image = self.transform(image)
        label = {"index": index, "class_label": label}
        if self.vlm_caption is not None:
            label["vlm_caption"] = self.vlm_caption.iloc[index]["caption"]
        return image, label


class ImageDataset(Dataset):
    def __init__(
        self,
        data_dirs: str | list[str],
        splits: Optional[str | list[Optional[str]]] = None,
        transform: Optional[Callable] = None,
        suffix=(".jpg", ".JPEG", ".png"),
        pil=True,
        return_dict=True,
    ) -> None:
        super().__init__()

        self.data_dirs = [data_dirs] if isinstance(data_dirs, str) else data_dirs
        if isinstance(splits, list):
            assert len(splits) == len(self.data_dirs)
            self.splits = splits
        elif isinstance(splits, str):
            assert len(self.data_dirs) == 1
            self.splits = [splits]
        else:
            self.splits = [None for _ in range(len(self.data_dirs))]

        self.transform = transform
        self.pil = pil
        self.return_dict = return_dict

        # load all images [image_path]
        self.samples = []
        for data_dir, split in zip(self.data_dirs, self.splits):
            if split is None:
                samples = load_image_from_dir(data_dir, suffix, return_mode="path")
            else:
                samples = []
                with open(split, "r") as fin:
                    for line in fin.readlines():
                        relative_path = line[:-1]
                        full_path = os.path.join(data_dir, relative_path)
                        samples.append(full_path)
            self.samples += samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int, skip_image=False) -> dict[str, Any]:
        image_path = self.samples[index]

        if skip_image:
            image = None
        else:
            image = load_image(image_path)
            if self.transform is not None:
                image = self.transform(image)
        if self.return_dict:
            return {
                "index": index,
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "data": image,
            }
        else:
            return image, 0


class MultiResolutionImageDataset(ImageDataset):
    def __init__(
        self,
        data_dirs: str | list[str],
        splits: Optional[str | list[Optional[str]]] = None,
        size_transform: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        suffix=(".jpg", ".JPEG", ".png"),
        pil=True,
        return_dict=True,
        vlm_caption: Optional[pandas.DataFrame] = None,
    ) -> None:
        super().__init__(data_dirs, splits, transform, suffix, pil, return_dict)
        self.size_transform = size_transform
        self.vlm_caption = vlm_caption

    def __getitem__(
        self, index: int | tuple[int, int] | tuple[int, int, int]
    ) -> tuple[torch.Tensor, Any] | dict[str, Any]:
        index, resolution, seed = parse_index(index)
        image_path = self.samples[index]
        image = np.load(image_path) if image_path.endswith(".npy") else load_image(image_path)
        image = self.size_transform(image, resolution, seed)
        if self.transform is not None:
            image = self.transform(image)
        if self.return_dict:
            return {
                "index": index,
                "image_path": image_path,
                "image_name": os.path.basename(image_path),
                "data": image,
            }
        else:
            label = {"index": index}
            if self.vlm_caption is not None:
                label["vlm_caption"] = self.vlm_caption.iloc[index]["caption"]
            return image, label


class MultiResolutionSubset(Subset):
    def __getitem__(
        self, index: int | tuple[int, int] | tuple[int, int, int]
    ) -> tuple[torch.Tensor, Any] | dict[str, Any]:
        if isinstance(index, int):
            return self.dataset[self.indices[index]]
        elif isinstance(index, tuple) and len(index) == 2:
            index, resolution = index
            return self.dataset[self.indices[index], resolution]
        elif isinstance(index, tuple) and len(index) == 3:
            index, resolution, seed = index
            return self.dataset[self.indices[index], resolution, seed]
        else:
            raise ValueError(f"index {index} is not supported")

    def __getitems__(self, indices: list[int] | list[tuple[int, int]] | list[tuple[int, int, int]]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if isinstance(indices[0], int):
            if callable(getattr(self.dataset, "__getitems__", None)):
                return self.dataset.__getitems__([self.indices[idx] for idx in indices])  # type: ignore[attr-defined]
            else:
                return [self.dataset[self.indices[idx]] for idx in indices]
        elif isinstance(indices[0], tuple) and len(indices[0]) == 2:
            if callable(getattr(self.dataset, "__getitems__", None)):
                return self.dataset.__getitems__([(self.indices[idx], resolution) for idx, resolution in indices])  # type: ignore[attr-defined]
            else:
                return [self.dataset[self.indices[idx], resolution] for idx, resolution in indices]
        elif isinstance(indices[0], tuple) and len(indices[0]) == 3:
            if callable(getattr(self.dataset, "__getitems__", None)):
                return self.dataset.__getitems__([(self.indices[idx], resolution, seed) for idx, resolution, seed in indices])  # type: ignore[attr-defined]
            else:
                return [self.dataset[self.indices[idx], resolution, seed] for idx, resolution, seed in indices]
        else:
            raise ValueError(f"indices {indices} is not supported")
