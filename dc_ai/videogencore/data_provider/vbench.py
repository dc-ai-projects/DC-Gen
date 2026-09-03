# Modified from the VBench & VBench 2.0 repository.
# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.

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

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from torch.utils.data import Dataset

from .base_eval import VideoGenCoreEvalDataProvider, VideoGenCoreEvalDataProviderConfig


@dataclass
class VBenchTextPromptDataProviderConfig(VideoGenCoreEvalDataProviderConfig):
    name: str = "VBenchTextPrompt"
    meta_path: str = "assets/data/vbench/VBench_extended_full_info.json"
    category_id: int = 0
    num_samples: Optional[int] = None
    num_videos_per_prompt: int = 5


@dataclass
class VBenchImagePromptDataProviderConfig(VideoGenCoreEvalDataProviderConfig):
    name: str = "VBenchImagePrompt"
    meta_path: str = "assets/data/vbench/VBench_i2v_extended_full_info.json"
    category_id: int = 0
    num_samples: Optional[int] = None
    num_videos_per_prompt: int = 5
    image_folder: str = "assets/data/vbench/vbench_i2v_imgs/832-480"


class VBenchTextPromptDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        category_id: int,
        num_samples: Optional[int] = None,
        seed: int = 0,
        num_videos_per_prompt: int = 5,
    ):
        super().__init__()
        if 0 <= category_id and category_id <= 10:
            with open(json_path, "r") as json_file:
                self.samples = list(json.load(json_file)[str(category_id)]["videos"])
            new_list = []
            for sample in self.samples:
                new_sample = copy.deepcopy(sample)
                new_sample["video_path"] = f"{sample['prompt']}.mp4"
                new_sample.pop("video_paths")
                new_list.append(new_sample)
            self.samples = new_list
            self.dimension_list = self.samples[0]["dimension"]
        else:
            self.samples = [
                {
                    "video_path": "astronaut_riding.mp4",
                    "dimension": "test",
                    "prompt": "astronaut is riding a horse on the moon, wearing a space suit and helmet. the horse is galloping across the lunar surface, leaving behind a trail of moon dust. in the background, earth is visible in the black sky, a beautiful blue and green marble. the astronaut is holding a flag with a logo on it, waving it proudly as they ride. the scene is surreal and whimsical, capturing the imagination of space exploration and adventure.",
                },
                {
                    "video_path": "magical_guru.mp4",
                    "dimension": "test",
                    "prompt": "soft lighting and warm colors infuse the image, creating a magical and serene effect. the view captures the serene guru man levitating above the golden sands, his long, flowing beard and simple robes gently swaying. his eyes are closed, and a peaceful smile graces his calm face. a gentle glow surrounds him, enhancing his aura of tranquility. behind him, the majestic pyramids of egypt loom, bathed in the warm light of the setting sun. the softly glowing sands and the pyramid silhouettes create a composition rich with spirituality and enlightenment, exuding an atmosphere of profound calmness.",
                },
                {
                    "video_path": "cherry_blossom.mp4",
                    "dimension": "test",
                    "prompt": "a soft-focus view captures a serene garden, filled with cherry blossom trees in bloom. at the center stands a beautiful japanese woman, portrayed in exquisite detail, wearing a traditional kimono with intricate floral patterns in soft pastel colors. her long, dark hair cascades elegantly down her back, enhancing her gentle, serene expression. pink petals drift down in a light breeze, adding to the garden's ethereal ambiance. sunlight filters through the leafy canopy, casting dappled shadows that dance around her, subtly highlighting her serene posture and the details of her kimono. the atmosphere is tranquil and picturesque, enveloped in a sense of timeless beauty.",
                },
                {
                    "video_path": "modern_cafe.mp4",
                    "dimension": "test",
                    "prompt": "the angle is mid-range, focusing on the well-dressed asian male friend sitting comfortably in a modern and stylish cafe. the setting exudes warmth and elegance, with soft music playing in the background to create an inviting atmosphere. the man holds a small gift box in his hand, his face illuminated by a confident smile as he looks around, his expression full of anticipation. his attire, a blend of classic and contemporary style, complements the chic surroundings, enhancing his poised demeanor. the ambient lighting accentuates his features, making the scene lively and intimate.",
                },
                {
                    "video_path": "training_stadium.mp4",
                    "dimension": "test",
                    "prompt": "the setting is a training facility, brightly lit with mirrored walls and sprung wooden floors. the scene starts with the camera panning slowly over the room, capturing the boundless determination in action as a group of korean girls rigorously practice their moves. each one is focused, their expressions marked by the intensity of a long and arduous journey. they are seen rehearsing complex choreography, with synchronized steps and practiced precision. alongside the strenuous physical training, snippets of their vocal lessons are interwoven, illustrating the multifaceted preparation involved. the atmosphere is one of dedication and discipline, reflected in their commitment to rigorous exercise and strict dietary habits. these scenes provide a glimpse into the grueling yet passionate pursuit of their dreams.",
                },
                {
                    "video_path": "puppies_snow.mp4",
                    "dimension": "test",
                    "prompt": "A litter of golden retriever puppies playing in the snow, their heads popping out covered in snowflakes. They tumble and play joyfully, wagging tails and bouncy movements. Snow-covered ground, close-up playful interaction.",
                },
                {
                    "video_path": "five_pups.mp4",
                    "dimension": "test",
                    "prompt": "Five gray wolf pups playfully frolic and chase each other along a remote gravel road, leaping and nipping at one another amidst tall grass. Medium shot, dynamic camera movement capturing their lively interactions.",
                },
                {
                    "video_path": "chair_excavate.mp4",
                    "dimension": "test",
                    "prompt": "Archeologists carefully excavate and gently dust off a generic plastic chair discovered in the desert. Wide shot showing the process, emphasizing their cautious movements and the dry, expansive desert landscape.",
                },
                {
                    "video_path": "cinema_display.mp4",
                    "dimension": "test",
                    "prompt": "A rotating camera view around a large stack of vintage TVs displaying various content like 1950s sci-fi films, horror movies, news, static, and a 1970s sitcom, set within a grand New York museum gallery.",
                },
                {
                    "video_path": "cat_boxing.mp4",
                    "dimension": "test",
                    "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
                },
                {
                    "video_path": "cat_surf.mp4",
                    "dimension": "test",
                    "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
                },
                {
                    "video_path": "shanghai_river.mp4",
                    "dimension": "test",
                    "prompt": "Oil painting style, the Bund in Shanghai at sunset. Majestic buildings with golden hues reflecting on the calm Huangpu River. People walking leisurely on the waterfront promenade. Wide shot capturing the skyline and river.",
                },
                {
                    "video_path": "lake_view.mp4",
                    "dimension": "test",
                    "prompt": "A serene park bench under a lush tree, with a calm lake and gentle ripples in the background. Medium shot, capturing the tranquil scenery and soft sunlight filtering through the leaves.",
                },
                {
                    "video_path": "cat_dog.mp4",
                    "dimension": "test",
                    "prompt": "A front view of a cat on the right and a dog on the left, both looking towards the camera. Close-up shot focusing on their facial expressions and interaction.",
                },
                {
                    "video_path": "surfboat_ski.mp4",
                    "dimension": "test",
                    "prompt": "A surfboard mounted on skis from the front view, showing the sleek design and detail of both the surfboard and skis. Close-up shot.",
                },
                {
                    "video_path": "peaceful_alley.mp4",
                    "dimension": "test",
                    "prompt": "A peaceful, quiet alleyway at sunset, with soft golden light filtering through narrow gaps between tall buildings. Shadows stretch across cobblestone pavement. Medium shot, slow pan revealing graffiti-covered walls and rusted metal doors.",
                },
            ]
            self.dimension_list = ["test"]

        if num_samples is not None:
            num_samples = min(num_samples, len(self.samples))
            random_state = np.random.RandomState(seed)
            random_indices = random_state.choice(len(self.samples), size=num_samples, replace=False).tolist()
            self.samples = [self.samples[i] for i in random_indices]
        elif num_videos_per_prompt > 1:
            repeated = []
            for sample in self.samples:
                for k in range(num_videos_per_prompt):
                    s = copy.deepcopy(sample)
                    base = s["video_path"].removesuffix(".mp4")
                    s["video_path"] = f"{base}-{k}.mp4"
                    repeated.append(s)
            self.samples = repeated

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "index": index,
            "path": self.samples[index]["video_path"],
            "dimension": self.samples[index]["dimension"],
            "prompt": self.samples[index]["prompt"],
        }
        if "extended_prompt" in self.samples[index]:
            sample["extended_prompt"] = self.samples[index]["extended_prompt"]
        if "auxiliary_info" in self.samples[index]:
            sample["auxiliary_info"] = self.samples[index]["auxiliary_info"]
        return sample


class VBenchTextPromptDataProvider(VideoGenCoreEvalDataProvider):
    def __init__(self, cfg: VBenchTextPromptDataProviderConfig):
        super().__init__(cfg)
        self.cfg: VBenchTextPromptDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = VBenchTextPromptDataset(
            json_path=self.cfg.meta_path,
            category_id=self.cfg.category_id,
            num_samples=self.cfg.num_samples,
            num_videos_per_prompt=self.cfg.num_videos_per_prompt,
        )
        self.dimension_list = dataset.dimension_list
        return dataset


class VBenchImagePromptDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        category_id: int,
        num_samples: Optional[int] = None,
        seed: int = 0,
        image_folder: str = "assets/data/vbench/vbench_i2v_imgs/832-480",
        num_videos_per_prompt: int = 5,
    ):
        super().__init__()
        if not (0 <= category_id and category_id <= 2):
            with open(json_path, "r") as json_file:
                self.samples = list(json.load(json_file)["0"]["videos"])
        else:
            with open(json_path, "r") as json_file:
                self.samples = list(json.load(json_file)[str(category_id)]["videos"])

        self.samples = [sample for sample in self.samples for _ in range(num_videos_per_prompt)]
        new_list = []
        for idx, sample in enumerate(self.samples):
            new_sample = copy.deepcopy(sample)
            new_sample["video_path"] = f"{sample['prompt']}-{idx%num_videos_per_prompt}.mp4"
            new_sample["image_path"] = os.path.join(image_folder, sample["image_name"])
            new_sample.pop("video_paths")
            new_sample.pop("image_name")
            new_list.append(new_sample)
        self.samples = new_list
        self.dimension_list = self.samples[0]["dimension"]

        if not (0 <= category_id and category_id <= 2):
            if num_samples is None:
                num_samples = 128
            else:
                num_samples = min(num_samples, 128)

        if num_samples is not None:
            num_samples = min(num_samples, len(self.samples))
            random_state = np.random.RandomState(seed)
            random_indices = random_state.choice(len(self.samples), size=num_samples, replace=False).tolist()
            self.samples = [self.samples[i] for i in random_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "index": index,
            "path": self.samples[index]["video_path"],
            "dimension": self.samples[index]["dimension"],
            "prompt": self.samples[index]["prompt"],
            "image_path": self.samples[index]["image_path"],
        }
        if "extended_prompt" in self.samples[index]:
            sample["extended_prompt"] = self.samples[index]["extended_prompt"]
        return sample


class VBenchImagePromptDataProvider(VideoGenCoreEvalDataProvider):
    def __init__(self, cfg: VBenchImagePromptDataProviderConfig):
        super().__init__(cfg)
        self.cfg: VBenchImagePromptDataProviderConfig

    def build_complete_dataset(self) -> Dataset:
        dataset = VBenchImagePromptDataset(
            json_path=self.cfg.meta_path,
            category_id=self.cfg.category_id,
            num_samples=self.cfg.num_samples,
            image_folder=self.cfg.image_folder,
            num_videos_per_prompt=self.cfg.num_videos_per_prompt,
        )
        self.dimension_list = dataset.dimension_list
        return dataset
