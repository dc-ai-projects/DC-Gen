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


from dataclasses import dataclass
from typing import Any

from omegaconf import MISSING

from .config import get_config


class BaseAspectRatioManager:
    def __init__(self):
        self.aspect_ratios = self._get_aspect_ratios()

    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        """Return the aspect ratio dictionary."""
        raise NotImplementedError

    def get_closest_ratio(self, height: int, width: int) -> str:
        """Get the closest aspect ratio for given height and width."""
        ratio = height / width
        return min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))

    def get_dimensions(self, ratio: str) -> tuple[int, int]:
        """Get the dimensions for a given aspect ratio."""
        return self.aspect_ratios[ratio]


class AspectRatioManager512F32MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (256, 1024),
            "0.27": (256, 960),
            "0.38": (320, 832),
            "0.42": (320, 768),
            "0.46": (352, 768),
            "0.48": (352, 736),
            "0.55": (384, 704),
            "0.57": (384, 672),
            "0.6": (384, 640),
            "0.65": (416, 640),
            "0.68": (416, 608),
            "0.74": (448, 608),
            "0.78": (448, 576),
            "0.83": (480, 576),
            "0.88": (480, 544),
            "1.0": (512, 512),
            "1.06": (544, 512),
            "1.2": (576, 480),
            "1.29": (576, 448),
            "1.36": (608, 448),
            "1.46": (608, 416),
            "1.62": (672, 416),
            "1.67": (640, 384),
            "2.4": (768, 320),
            "3.75": (960, 256),
            "4.0": (1024, 256),
        }


class AspectRatioManager512F64MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (256, 1024),
            "0.27": (256, 960),
            "0.42": (320, 768),
            "0.6": (384, 640),
            "0.78": (448, 576),
            "1.0": (512, 512),
            "1.29": (576, 448),
            "1.67": (640, 384),
            "2.4": (768, 320),
            "3.75": (960, 256),
            "4.0": (1024, 256),
        }


AspectRatioManager512 = AspectRatioManager512F64MS


class AspectRatioManager1024(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (512, 2048),
            "0.26": (512, 1984),
            "0.27": (512, 1920),
            "0.28": (512, 1856),
            "0.32": (576, 1792),
            "0.33": (576, 1728),
            "0.35": (576, 1664),
            "0.4": (640, 1600),
            "0.42": (640, 1536),
            "0.48": (704, 1472),
            "0.5": (704, 1408),
            "0.52": (704, 1344),
            "0.57": (768, 1344),
            "0.6": (768, 1280),
            "0.68": (832, 1216),
            "0.72": (832, 1152),
            "0.78": (896, 1152),
            "0.82": (896, 1088),
            "0.88": (960, 1088),
            "0.94": (960, 1024),
            "1.0": (1024, 1024),
            "1.07": (1024, 960),
            "1.13": (1088, 960),
            "1.21": (1088, 896),
            "1.29": (1152, 896),
            "1.38": (1152, 832),
            "1.46": (1216, 832),
            "1.67": (1280, 768),
            "1.75": (1344, 768),
            "2.0": (1408, 704),
            "2.09": (1472, 704),
            "2.4": (1536, 640),
            "2.5": (1600, 640),
            "2.89": (1664, 576),
            "3.0": (1728, 576),
            "3.11": (1792, 576),
            "3.62": (1856, 512),
            "3.75": (1920, 512),
            "3.88": (1984, 512),
            "4.0": (2048, 512),
        }


class AspectRatioManager2048(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (1024, 4096),
            "0.26": (1024, 3968),
            "0.27": (1024, 3840),
            "0.28": (1024, 3712),
            "0.32": (1152, 3584),
            "0.33": (1152, 3456),
            "0.35": (1152, 3328),
            "0.4": (1280, 3200),
            "0.42": (1280, 3072),
            "0.48": (1408, 2944),
            "0.5": (1408, 2816),
            "0.52": (1408, 2688),
            "0.57": (1536, 2688),
            "0.6": (1536, 2560),
            "0.68": (1664, 2432),
            "0.72": (1664, 2304),
            "0.78": (1792, 2304),
            "0.82": (1792, 2176),
            "0.88": (1920, 2176),
            "0.94": (1920, 2048),
            "1.0": (2048, 2048),
            "1.07": (2048, 1920),
            "1.13": (2176, 1920),
            "1.21": (2176, 1792),
            "1.29": (2304, 1792),
            "1.38": (2304, 1664),
            "1.46": (2432, 1664),
            "1.67": (2560, 1536),
            "1.75": (2688, 1536),
            "2.0": (2816, 1408),
            "2.09": (2944, 1408),
            "2.4": (3072, 1280),
            "2.5": (3200, 1280),
            "2.89": (3328, 1152),
            "3.0": (3456, 1152),
            "3.11": (3584, 1152),
            "3.62": (3712, 1024),
            "3.75": (3840, 1024),
            "3.88": (3968, 1024),
            "4.0": (4096, 1024),
        }


class AspectRatioManager4096(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (2048, 8192),
            "0.26": (2048, 7936),
            "0.27": (2048, 7680),
            "0.28": (2048, 7424),
            "0.32": (2304, 7168),
            "0.33": (2304, 6912),
            "0.35": (2304, 6656),
            "0.4": (2560, 6400),
            "0.42": (2560, 6144),
            "0.48": (2816, 5888),
            "0.5": (2816, 5632),
            "0.52": (2816, 5376),
            "0.57": (3072, 5376),
            "0.6": (3072, 5120),
            "0.68": (3328, 4864),
            "0.72": (3328, 4608),
            "0.78": (3584, 4608),
            "0.82": (3584, 4352),
            "0.88": (3840, 4352),
            "0.94": (3840, 4096),
            "1.0": (4096, 4096),
            "1.07": (4096, 3840),
            "1.13": (4352, 3840),
            "1.21": (4352, 3584),
            "1.29": (4608, 3584),
            "1.38": (4608, 3328),
            "1.46": (4864, 3328),
            "1.67": (5120, 3072),
            "1.75": (5376, 3072),
            "2.0": (5632, 2816),
            "2.09": (5888, 2816),
            "2.4": (6144, 2560),
            "2.5": (6400, 2560),
            "2.89": (6656, 2304),
            "3.0": (6912, 2304),
            "3.11": (7168, 2304),
            "3.62": (7424, 2048),
            "3.75": (7680, 2048),
            "3.88": (7936, 2048),
            "4.0": (8192, 2048),
        }


class AspectRatioManagerVideo480F32MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.5": (448, 896),
            "0.5625": (480, 832),
            "0.68": (512, 768),
            "0.78": (544, 704),
            "1.0": (640, 640),
            "1.13": (672, 608),
            "1.29": (704, 544),
            "1.46": (768, 512),
            "1.67": (832, 480),
            "1.75": (832, 480),
            "2.0": (896, 448),
        }


class AspectRatioManagerVideo480F64MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.5": (448, 896),
            "0.5625": (448, 832),
            "0.68": (512, 768),
            "0.78": (576, 704),
            "1.0": (640, 640),
            "1.13": (640, 576),
            "1.29": (704, 576),
            "1.46": (768, 512),
            "1.67": (832, 512),
            "1.75": (832, 448),
            "2.0": (896, 448),
        }


class AspectRatioManagerVideo720(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.57": (720, 1280),
        }


class AspectRatioManagerVideo720F32MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.5": (672, 1344),
            "0.5625": (704, 1280),
            "0.68": (800, 1152),
            "0.78": (832, 1088),
            "1.0": (960, 960),
            "1.13": (1024, 896),
            "1.29": (1088, 832),
            "1.46": (1152, 800),
            "1.67": (1248, 736),
            "1.75": (1280, 736),
            "2.0": (1344, 672),
        }


class AspectRatioManagerVideo720F64MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.5": (704, 1344),
            "0.5625": (704, 1280),
            "0.68": (768, 1152),
            "0.78": (832, 1088),
            "1.0": (960, 960),
            "1.13": (1024, 896),
            "1.29": (1088, 832),
            "1.46": (1152, 768),
            "1.67": (1216, 768),
            "1.75": (1280, 704),
            "2.0": (1344, 704),
        }


class AspectRatioManagerVideo1080F32MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (704, 2880),
            "0.42": (928, 2208),
            "0.5625": (1088, 1920),
            "0.81": (1280, 1600),
            "0.94": (1408, 1472),
            "1.0": (1440, 1440),
            "1.06": (1472, 1408),
            "1.23": (1600, 1312),
            "1.67": (1856, 1120),
            "2.4": (2240, 928),
            "4.0": (2880, 704),
        }


class AspectRatioManagerVideo1080F64MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (704, 2880),
            "0.42": (960, 2240),
            "0.5625": (1088, 1920),
            "0.81": (1280, 1600),
            "0.94": (1408, 1472),
            "1.0": (1408, 1408),
            "1.06": (1472, 1408),
            "1.23": (1600, 1280),
            "1.67": (1856, 1088),
            "2.4": (2240, 960),
            "4.0": (2880, 704),
        }


class AspectRatioManagerVideo2160F32MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (1440, 5760),
            "0.42": (1856, 4448),
            "0.5625": (2176, 3840),
            "0.81": (2592, 3200),
            "0.94": (2784, 2976),
            "1.0": (2880, 2880),
            "1.06": (2976, 2784),
            "1.23": (3200, 2592),
            "1.67": (3712, 2240),
            "2.4": (4448, 1856),
            "4.0": (5760, 1440),
        }


class AspectRatioManagerVideo2160F64MS(BaseAspectRatioManager):
    def _get_aspect_ratios(self) -> dict[str, tuple[int, int]]:
        return {
            "0.25": (1408, 5760),
            "0.42": (1856, 4416),
            "0.5625": (2176, 3840),
            "0.81": (2560, 3200),
            "0.94": (2816, 2944),
            "1.0": (2880, 2880),
            "1.06": (2944, 2816),
            "1.23": (3200, 2624),
            "1.67": (3712, 2240),
            "2.4": (4480, 1856),
            "4.0": (5760, 1408),
        }


def get_aspect_ratio_manager(resolution: str) -> BaseAspectRatioManager:
    if resolution == "480F64MS":
        return AspectRatioManagerVideo480F64MS()
    elif resolution == "480F32MS":
        return AspectRatioManagerVideo480F32MS()
    elif resolution == "720F64MS":
        return AspectRatioManagerVideo720F64MS()
    elif resolution == "720":
        return AspectRatioManagerVideo720()
    elif resolution == "720F32MS":
        return AspectRatioManagerVideo720F32MS()
    elif resolution == "1080F32MS":
        return AspectRatioManagerVideo1080F32MS()
    elif resolution == "1080F64MS":
        return AspectRatioManagerVideo1080F64MS()
    elif resolution == "2160F32MS":
        return AspectRatioManagerVideo2160F32MS()
    elif resolution == "2160F64MS":
        return AspectRatioManagerVideo2160F64MS()
    else:
        raise ValueError(f"resolution {resolution} is not supported")


@dataclass
class GenerateAspectRatioConfig:
    resolution: Any = MISSING
    spatial_compression_ratio: int = MISSING
    ratios: tuple[float, ...] = MISSING


def main():
    cfg = get_config(GenerateAspectRatioConfig)
    if isinstance(cfg.resolution, int):
        area = cfg.resolution * cfg.resolution
    elif (
        isinstance(cfg.resolution, list)
        and all(isinstance(item, int) for item in cfg.resolution)
        and len(cfg.resolution) == 2
    ):
        area = cfg.resolution[0] * cfg.resolution[1]
    else:
        raise ValueError(f"resolution {cfg.resolution} is not supported")

    for ratio in cfg.ratios:
        h = (area * ratio) ** 0.5
        w = (area / ratio) ** 0.5
        h = round(h / cfg.spatial_compression_ratio) * cfg.spatial_compression_ratio
        w = round(w / cfg.spatial_compression_ratio) * cfg.spatial_compression_ratio
        print(f'"{ratio}": ({h}, {w}),')


if __name__ == "__main__":
    main()


"""
python -m dc_ai.apps.utils.aspect_ratio resolution=[480,832] spatial_compression_ratio=32 ratios=[0.5,0.5625,0.68,0.78,1.0,1.13,1.29,1.46,1.67,1.75,2.0]
python -m dc_ai.apps.utils.aspect_ratio resolution=[480,832] spatial_compression_ratio=64 ratios=[0.5,0.5625,0.68,0.78,1.0,1.13,1.29,1.46,1.67,1.75,2.0]

python -m dc_ai.apps.utils.aspect_ratio resolution=[720,1280] spatial_compression_ratio=32 ratios=[0.5,0.5625,0.68,0.78,1.0,1.13,1.29,1.46,1.67,1.75,2.0]
python -m dc_ai.apps.utils.aspect_ratio resolution=[720,1280] spatial_compression_ratio=64 ratios=[0.5,0.5625,0.68,0.78,1.0,1.13,1.29,1.46,1.67,1.75,2.0]

python -m dc_ai.apps.utils.aspect_ratio resolution=[1080,1920] spatial_compression_ratio=32 ratios=[0.25,0.42,0.5625,0.81,0.94,1.0,1.06,1.23,1.67,2.4,4.0]
python -m dc_ai.apps.utils.aspect_ratio resolution=[1080,1920] spatial_compression_ratio=64 ratios=[0.25,0.42,0.5625,0.81,0.94,1.0,1.06,1.23,1.67,2.4,4.0]

python -m dc_ai.apps.utils.aspect_ratio resolution=[2160,3840] spatial_compression_ratio=32 ratios=[0.25,0.42,0.5625,0.81,0.94,1.0,1.06,1.23,1.67,2.4,4.0]
python -m dc_ai.apps.utils.aspect_ratio resolution=[2160,3840] spatial_compression_ratio=64 ratios=[0.25,0.42,0.5625,0.81,0.94,1.0,1.06,1.23,1.67,2.4,4.0]
"""
