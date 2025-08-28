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
from typing import Any, Dict, List, Tuple

import ipdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def generate_tar(data_format: Dict[str, str], data_list: List[Tuple[str, Dict[str, Any]]], tar_path: str) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    tar_path = os.path.abspath(tar_path)
    tar = tarfile.open(tar_path, "w")
    for prefix, data in tqdm(data_list, desc=f"generate {tar_path}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"):
        for key in data_format:
            if data_format[key] == "file_path":
                tar.add(data[key], arcname=prefix + key)
            elif data_format[key] == "data":
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
                elif key == ".pth":
                    fileobj = io.BytesIO()
                    torch.save(data[key], fileobj)
                    fileobj.seek(0)
                else:
                    raise ValueError(f"{key} is not supported as raw data")
                tar_info = tarfile.TarInfo(name=prefix + key)
                tar_info.size = fileobj.getbuffer().nbytes
                tar_info.mtime = int(time.time())  # avoids large header size
                tar_info.uname = getpass.getuser()
                tar_info.gname = "dip"
                tar.addfile(tar_info, fileobj)
            else:
                raise ValueError(f"data format for {key} {data_format[key]} is not supported")

    tar.close()


def generate_all_tar(
    data_format: Dict[str, str],
    data_list: List[Tuple[str, Dict[str, Any]]],
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
    def __init__(self, tar_path: str):
        with tarfile.open(tar_path, "r") as tar_file:
            self.tar_stream = open(tar_path, "rb")
            self.sample_prefix_list = []
            self.sample_meta_list = []
            last_prefix = ""
            self.mmapped_file = mmap.mmap(
                self.tar_stream.fileno(), 0, access=mmap.ACCESS_READ
            )  # mmap is necessary since tarfile doesn't work with multiprocessing
            for tarinfo in tar_file:
                prefix, ext = os.path.splitext(tarinfo.name)
                if prefix != last_prefix:
                    self.sample_meta_list.append({})
                    self.sample_prefix_list.append(prefix)
                self.sample_meta_list[-1][ext] = (tarinfo.name, tarinfo.size, tar_file.fileobj.tell())
                last_prefix = prefix

    def __len__(self):
        return len(self.sample_meta_list)

    def __getitem__(self, index: int):
        sample = {"__key__": self.sample_prefix_list[index]}
        for ext in self.sample_meta_list[index]:
            name, size, offset = self.sample_meta_list[index][ext]
            stream = io.BytesIO(self.mmapped_file[offset : offset + size])
            if ext == ".json":
                sample[ext] = json.load(stream)
            elif ext in [".jpg", ".jpeg", ".png", ".ppm", ".pgm", ".pbm", ".pnm", ".webp", ".bmp", ".tiff"]:
                sample[ext] = Image.open(stream)
            elif ext == ".npy":
                sample[ext] = np.load(stream)
            elif ext == ".pth":
                sample[ext] = torch.load(stream)
            else:
                raise ValueError(f"Unsupported ext: {ext}")
        return sample

    def __del__(self):
        self.mmapped_file.close()
        self.tar_stream.close()
