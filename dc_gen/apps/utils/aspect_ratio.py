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


class AspectRatioManager512(BaseAspectRatioManager):
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
