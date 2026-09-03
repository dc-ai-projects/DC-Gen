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

import io
import json
import os
import random
import zipfile
from multiprocessing import Pool
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ..data_provider.zip_dataset.single_zip_dataset import SingleZipDataset, single_zip_dataset_worker_init_fn

__all__ = ["generate_zip", "SingleZipDataset", "single_zip_dataset_worker_init_fn"]


def generate_zip(data_format: dict[str, str], data_list: list[tuple[str, dict[str, Any]]], zip_path: str) -> None:
    """
    data_format: `key` is suffix like ".png", `value` should be str from ["file_path", "data"], "file_path" means `data[key]` is a file path, "data" means `data[key]` is raw data.
    data_list: List of (prefix, data). For each data, `key` should be in `data_format`, `data[key]` should be either a file path or raw data.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for prefix, data in tqdm(
            data_list, desc=f"Zipping to {zip_path}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
        ):
            for suffix, content in data.items():
                arcname = prefix + suffix

                if data_format.get(suffix) == "file_path":
                    zf.write(content, arcname=arcname)
                elif data_format.get(suffix) == "data":
                    file_bytes = None
                    if suffix == ".json":
                        file_bytes = json.dumps(content, indent=None, separators=(",", ":")).encode("utf-8")
                    elif suffix == ".npy":
                        with io.BytesIO() as fileobj:
                            np.save(fileobj, content)
                            file_bytes = fileobj.getvalue()
                    elif suffix == ".jpg" or suffix == ".jpeg":
                        if isinstance(content, Image.Image):
                            with io.BytesIO() as fileobj:
                                content.save(fileobj, format="JPEG")
                                file_bytes = fileobj.getvalue()
                        elif isinstance(content, io.BytesIO):
                            file_bytes = content.getvalue()
                        else:
                            raise ValueError(f"type {type(content)} is not supported for .jpg")
                    elif suffix == ".png":
                        if isinstance(content, Image.Image):
                            with io.BytesIO() as fileobj:
                                content.save(fileobj, format="PNG")
                                file_bytes = fileobj.getvalue()
                        elif isinstance(content, io.BytesIO):
                            file_bytes = content.getvalue()
                        else:
                            raise ValueError(f"type {type(content)} is not supported for .png")
                    elif suffix == ".pt" or suffix == ".pth":
                        with io.BytesIO() as fileobj:
                            torch.save(content, fileobj)
                            file_bytes = fileobj.getvalue()
                    elif suffix == ".txt":
                        file_bytes = str(content).encode("utf-8")
                    else:
                        raise ValueError(f"Unsupported suffix for generate zip: {suffix}")

                    zf.writestr(arcname, file_bytes)
                else:
                    raise ValueError(f"data format for {suffix} {data_format.get(suffix)} is not supported")


def generate_all_zip(
    data_format: dict[str, str],
    data_list: list[tuple[str, dict[str, Any]]],
    output_dir: str,
    num_samples_per_archive: int,
    shuffle: bool,
    num_digits_in_archive_name: int,
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
    os.makedirs(output_dir, exist_ok=True)
    num_samples = len(data_list)

    if processes == 1:
        for archive_idx, start in enumerate(range(0, num_samples, num_samples_per_archive)):
            generate_zip(
                data_format,
                data_list[start : min(start + num_samples_per_archive, num_samples)],
                os.path.join(output_dir, f"{archive_idx:0{num_digits_in_archive_name}d}.tar"),
            )
    else:
        pool = Pool(processes=processes)
        for archive_idx, start in enumerate(range(0, num_samples, num_samples_per_archive)):
            pool.apply_async(
                generate_zip,
                args=(
                    data_format,
                    data_list[start : min(start + num_samples_per_archive, num_samples)],
                    os.path.join(output_dir, f"{archive_idx:0{num_digits_in_archive_name}d}.zip"),
                ),
            )
        pool.close()
        pool.join()
