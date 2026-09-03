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

import getpass
import io
import json
import mmap
import os
import random
import tarfile
import time
from multiprocessing import Pool
from typing import Any, Optional

import ipdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .video import VideoLoader, write_video


class TarWriter:
    def __init__(self, tar_path: str, data_format: dict[str, str]):
        """
        data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
        """
        tar_path = os.path.abspath(tar_path)
        self.tar = tarfile.open(tar_path, "w")
        self.data_format = data_format

    def add_data(self, prefix: str, data: dict[str, Any]):
        """
        For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
        """
        for key in self.data_format:
            if self.data_format[key] == "file_path":
                self.tar.add(data[key], arcname=prefix + key)
            elif self.data_format[key] == "data":
                if key == ".json":
                    fileobj = io.BytesIO(json.dumps(data[key]).encode("utf-8"))
                elif key == ".jpg":
                    if isinstance(data[key], Image.Image):
                        fileobj = io.BytesIO()
                        data[key].save(fileobj, format="JPEG")
                        fileobj.seek(0)
                    elif isinstance(data[key], io.BytesIO):
                        fileobj = data[key]
                    else:
                        raise ValueError(f"type {type(data[key])} is not supported for .jpg")
                elif key == ".npy":
                    fileobj = io.BytesIO()
                    np.save(fileobj, data[key])
                    fileobj.seek(0)
                elif key == ".npz":
                    fileobj = io.BytesIO()
                    np.savez(fileobj, **data[key])
                    fileobj.seek(0)
                elif key == ".pth":
                    fileobj = io.BytesIO()
                    torch.save(data[key], fileobj)
                    fileobj.seek(0)
                elif key == ".mp4":
                    fileobj = io.BytesIO()
                    write_video(fileobj, images=data[key][0], fps=data[key][1])
                    fileobj.seek(0)
                else:
                    raise ValueError(f"{key} is not supported as raw data")
                tar_info = tarfile.TarInfo(name=prefix + key)
                tar_info.size = fileobj.getbuffer().nbytes
                tar_info.mtime = int(time.time())  # avoids large header size
                tar_info.uname = getpass.getuser()
                tar_info.gname = "dip"
                self.tar.addfile(tar_info, fileobj)
            else:
                raise ValueError(f"data format for {key} {self.data_format[key]} is not supported")

    def close(self):
        self.tar.close()


def generate_tar(data_format: dict[str, str], data_list: list[tuple[str, dict[str, Any]]], tar_path: str) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    tar_writer = TarWriter(tar_path, data_format)
    for prefix, data in tqdm(data_list, desc=f"generate {tar_path}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        tar_writer.add_data(prefix, data)
    tar_writer.close()


def generate_all_tar(
    data_format: dict[str, str],
    data_list: list[tuple[str, dict[str, Any]]],
    tar_dir: str,
    num_samples_per_tar: int,
    shuffle: bool,
    num_digits_in_tar_name: int,
    processes: int,
    seed: int = 0,
) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    if shuffle:
        random.seed(seed)
        random.shuffle(data_list)
    os.makedirs(tar_dir, exist_ok=True)
    num_samples = len(data_list)

    if processes == 1:
        for tar_idx, start in enumerate(range(0, num_samples, num_samples_per_tar)):
            generate_tar(
                data_format,
                data_list[start : min(start + num_samples_per_tar, num_samples)],
                os.path.join(tar_dir, f"{tar_idx:0{num_digits_in_tar_name}d}.tar"),
            )
    else:
        pool = Pool(processes=processes)
        for tar_idx, start in enumerate(range(0, num_samples, num_samples_per_tar)):
            pool.apply_async(
                generate_tar,
                args=(
                    data_format,
                    data_list[start : min(start + num_samples_per_tar, num_samples)],
                    os.path.join(tar_dir, f"{tar_idx:0{num_digits_in_tar_name}d}.tar"),
                ),
            )
        pool.close()
        pool.join()


class SingleTarDataset(Dataset):
    def __init__(
        self, tar_path: Optional[str] = None, fileobj: Optional[io.IOBase] = None, *, return_raw_video: bool = False
    ):
        assert (tar_path is not None) + (fileobj is not None) == 1
        self.tar_path = tar_path
        if tar_path is not None:
            self.tar_stream = open(tar_path, "rb")
            self._file_data = mmap.mmap(
                self.tar_stream.fileno(), 0, access=mmap.ACCESS_READ
            )  # mmap is necessary since tarfile doesn't work with multiprocessing
        elif fileobj is not None:
            self._file_data = fileobj.read()  # BytesIO has no fileno(); use bytes
            fileobj.seek(0)
        else:
            raise ValueError("tar_path and fileobj cannot be None at the same time")
        with tarfile.open(name=tar_path, fileobj=fileobj, mode="r") as tar_file:
            self.sample_prefix_list = []
            self.sample_meta_list = []
            last_prefix = ""
            for tarinfo in tar_file:
                prefix, ext = os.path.splitext(tarinfo.name)
                if prefix != last_prefix:
                    self.sample_meta_list.append({})
                    self.sample_prefix_list.append(prefix)
                self.sample_meta_list[-1][ext] = (tarinfo.name, tarinfo.size, tar_file.fileobj.tell())
                last_prefix = prefix
            self.key_to_index = {prefix: index for index, prefix in enumerate(self.sample_prefix_list)}
        self.return_raw_video = return_raw_video

    def set_return_raw_video(self, return_raw_video: bool):
        self.return_raw_video = return_raw_video

    def __len__(self):
        return len(self.sample_meta_list)

    def __getitem__(self, index: int):
        sample = {"__key__": self.sample_prefix_list[index]}
        for ext in self.sample_meta_list[index]:
            name, size, offset = self.sample_meta_list[index][ext]
            stream = io.BytesIO(self._file_data[offset : offset + size])
            if ext == ".json":
                try:
                    sample[ext] = json.load(stream)
                except json.decoder.JSONDecodeError:
                    print(f"Error loading json from {name}")
                    sample[ext] = {}
            elif ext in [".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pbm", ".pnm", ".webp", ".bmp", ".tiff"]:
                sample[ext] = Image.open(stream)
            elif ext == ".npy":
                sample[ext] = np.load(stream)
            elif ext == ".npz":
                npz_data = np.load(stream)
                sample[ext] = {name: npz_data[name] for name in npz_data.files}
            elif ext == ".mp4":
                sample[ext] = VideoLoader(stream) if not self.return_raw_video else stream
            elif ext == ".pth":
                sample[ext] = torch.load(stream)
            else:
                raise ValueError(f"Unsupported ext: {ext}")
        return sample

    def get_item_by_key(self, key: str):
        return self[self.key_to_index[key]]

    def __del__(self):
        if self.tar_path is not None:
            self._file_data.close()
        if hasattr(self, "tar_stream"):
            self.tar_stream.close()
