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

import json
import pickle
from io import BytesIO
from typing import Any

import numpy as np
import PIL.Image
import torch

from ..utils.video import VideoLoader


def decode_bytes(file_bytes: bytes, extension: str) -> Any:
    """
    Decode bytes based on the file extension.
    Args:
        file_bytes (bytes): the bytes to decode.
        extension (str): extension of file, like '.npy', '.json', '.txt'。

    Returns:
        Any: decoded data, which can be a numpy array, a dictionary, or a string.
    """
    stream = BytesIO(file_bytes)
    extension = extension.lower().lstrip(".")
    if extension == "npy":
        return np.load(stream, allow_pickle=True)
    elif extension == "json":
        return json.load(stream)
    elif extension == "txt" or extension == "text":
        return file_bytes.decode("utf-8")
    elif extension in ["jpg", "jpeg", "png", "ppm", "pgm", "pbm", "pnm", "webp", "bmp", "tiff"]:
        return PIL.Image.open(stream)
    elif extension == "mp4":
        return VideoLoader(stream)
    elif extension in ["pt", "pth"]:
        return torch.load(stream, map_location="cpu")
    elif extension in ["pickle", "pkl"]:
        return pickle.loads(file_bytes)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")
