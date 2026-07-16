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
from collections import defaultdict
from typing import Any
from zipfile import ZipFile

import torch
from torch.utils.data import Dataset

from ..utils import decode_bytes


def single_zip_dataset_worker_init_fn(word_id: int):
    """
    Using SingleZipDataset without initializing the zip file handle in the constructor will cause issues when using multiple workers. Possibly due to the heavy I/O operations.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        dataset: SingleZipDataset = worker_info.dataset
        assert isinstance(dataset, SingleZipDataset), "Expected dataset to be an instance of SingleZipDataset"
        dataset.zip_file_handle = ZipFile(dataset.zip_path, "r")


class SingleZipDataset(Dataset):
    def __init__(self, zip_path: str):
        super().__init__()
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Zip file not found at {zip_path}")
        self.zip_path = zip_path

        # Initialize file handle to None
        # This will be set during the first access to the dataset
        self.zip_file_handle: ZipFile | None = None
        self.samples: list[tuple[str, list[str]]]
        self._load_metadata()

    def _load_metadata(self):
        key_to_files = defaultdict(list)
        with ZipFile(self.zip_path, "r") as zf:
            for filename in zf.namelist():
                key, _ = os.path.splitext(filename)
                key_to_files[key].append(filename)

        self.samples = []
        for key, value in key_to_files.items():
            self.samples.append((key, value))

    def _open_zip_file(self):
        """
        Initializes the file handle when each worker accesses it for the first time.
        """
        self.zip_file_handle = ZipFile(self.zip_path, "r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Args:
            index (int)

        Returns:
            Dict[str, Any]: sample dictionary with the following structure:
                            {'__key__': 'key', '.npy': ndarray, '.json': dict}
        """
        if self.zip_file_handle is None:
            self._open_zip_file()

        key, files_for_key = self.samples[index]
        sample_dict = {"__key__": key}

        for filename in files_for_key:
            _, extension = os.path.splitext(filename)
            file_bytes = self.zip_file_handle.read(filename)
            sample_dict[extension] = decode_bytes(file_bytes, extension)

        return sample_dict

    def __del__(self):
        if self.zip_file_handle:
            self.zip_file_handle.close()
