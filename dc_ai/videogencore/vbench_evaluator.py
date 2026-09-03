# Modified from the VBench & VBench 2.0 repository.
# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.

import torch
from PIL import Image
from tqdm import tqdm

from ..apps.utils.dist import is_master
from .data_provider.vbench import (
    VBenchImagePromptDataProvider,
    VBenchImagePromptDataProviderConfig,
    VBenchTextPromptDataProvider,
    VBenchTextPromptDataProviderConfig,
)
from .models.base import BaseVideoGenModel
from .trainer import VideoGenCoreTrainer


def build_metric_computer(dimension_list, device, submodules_list):
    computer_list = []
    key_list = []

    #  Quality Metrics
    if "subject_consistency" in dimension_list:
        from vbench.subject_consistency import ComputeSingleSubjectConsistency

        computer_list.append(ComputeSingleSubjectConsistency(device, submodules_list["subject_consistency"]))
        key_list.append("subject_consistency")
    if "motion_smoothness" in dimension_list:
        from vbench.motion_smoothness import ComputeSingleMotionSmoothness

        computer_list.append(ComputeSingleMotionSmoothness(device, submodules_list["motion_smoothness"]))
        key_list.append("motion_smoothness")
    if "dynamic_degree" in dimension_list:
        from vbench.dynamic_degree import ComputeSingleDynamicDegree

        computer_list.append(ComputeSingleDynamicDegree(device, submodules_list["dynamic_degree"]))
        key_list.append("dynamic_degree")
    if "aesthetic_quality" in dimension_list:
        from vbench.aesthetic_quality import ComputeSingleAestheticQuality

        computer_list.append(ComputeSingleAestheticQuality(device, submodules_list["aesthetic_quality"]))
        key_list.append("aesthetic_quality")
    if "imaging_quality" in dimension_list:
        from vbench.imaging_quality import ComputeSingleImagingQuality

        computer_list.append(ComputeSingleImagingQuality(device, submodules_list["imaging_quality"]))
        key_list.append("imaging_quality")
    if "temporal_flickering" in dimension_list:
        from vbench.temporal_flickering import ComputeSingleTemporalFlickering

        computer_list.append(ComputeSingleTemporalFlickering(device, submodules_list["temporal_flickering"]))
        key_list.append("temporal_flickering")
    if "background_consistency" in dimension_list:
        from vbench.background_consistency import ComputeSingleBackgroundConsistency

        computer_list.append(ComputeSingleBackgroundConsistency(device, submodules_list["background_consistency"]))
        key_list.append("background_consistency")

    #  Semantic Metrics
    if "overall_consistency" in dimension_list:
        from vbench.overall_consistency import ComputeSingleOverallConsistency

        computer_list.append(ComputeSingleOverallConsistency(device, submodules_list["overall_consistency"]))
        key_list.append("overall_consistency")
    if "temporal_style" in dimension_list:
        from vbench.temporal_style import ComputeSingleTemporalStyle

        computer_list.append(ComputeSingleTemporalStyle(device, submodules_list["temporal_style"]))
        key_list.append("temporal_style")
    if "human_action" in dimension_list:
        from vbench.human_action import ComputeSingleHumanAction

        computer_list.append(ComputeSingleHumanAction(device, submodules_list["human_action"]))
        key_list.append("human_action")
    if "object_class" in dimension_list:
        from vbench.object_class import ComputeSingleObjectClass

        computer_list.append(ComputeSingleObjectClass(device, submodules_list["object_class"]))
        key_list.append("object_class")
    if "multiple_objects" in dimension_list:
        from vbench.multiple_objects import ComputeSingleMultipleObjects

        computer_list.append(ComputeSingleMultipleObjects(device, submodules_list["multiple_objects"]))
        key_list.append("multiple_objects")
    if "color" in dimension_list:
        from vbench.color import ComputeSingleColor

        computer_list.append(ComputeSingleColor(device, submodules_list["color"]))
        key_list.append("color")
    if "spatial_relationship" in dimension_list:
        from vbench.spatial_relationship import ComputeSingleSpatialRelationship

        computer_list.append(ComputeSingleSpatialRelationship(device, submodules_list["spatial_relationship"]))
        key_list.append("spatial_relationship")
    if "scene" in dimension_list:
        from vbench.scene import ComputeSingleScene

        computer_list.append(ComputeSingleScene(device, submodules_list["scene"]))
        key_list.append("scene")
    if "appearance_style" in dimension_list:
        from vbench.appearance_style import ComputeSingleAppearanceStyle

        computer_list.append(ComputeSingleAppearanceStyle(device, submodules_list["appearance_style"]))
        key_list.append("appearance_style")

    #  I2V Metrics
    if "i2v_subject" in dimension_list:
        from vbench.i2v_subject import ComputeSingleI2VSubject

        computer_list.append(ComputeSingleI2VSubject(device, submodules_list["i2v_subject"]))
        key_list.append("i2v_subject")
    if "i2v_background" in dimension_list:
        from vbench.i2v_background import ComputeSingleI2VBackground

        computer_list.append(ComputeSingleI2VBackground(device, submodules_list["i2v_background"]))
        key_list.append("i2v_background")
    if "camera_motion" in dimension_list:
        from vbench.camera_motion import ComputeSingleCameraMotion

        computer_list.append(ComputeSingleCameraMotion(device, submodules_list["camera_motion"]))
        key_list.append("camera_motion")

    return computer_list, key_list


class VBenchEvaluator:
    METRICS_NORMALIZATION_RANGES = {
        "subject_consistency": [0.1462, 1.0],
        "motion_smoothness": [0.706, 0.9975],
        "temporal_flickering": [0.6293, 1.0],
        "background_consistency": [0.2615, 1.0],
        "scene": [0.0, 0.8222],
        "appearance_style": [0.0009, 0.2855],
        "temporal_style": [0.0, 0.364],
        "overall_consistency": [0.0, 0.364],
    }

    def __init__(
        self,
        trainer: VideoGenCoreTrainer,
        network: BaseVideoGenModel,
        batch_size,
        num_samples,
        num_videos_per_prompt,
        cfg_scale,
        pag_scale,
        eval_generator,
    ):
        self.cfg = trainer.cfg
        self.trainer = trainer
        self.network = network
        self.device = trainer.device
        self.batch_size, self.num_samples = batch_size, num_samples
        self.num_videos_per_prompt = num_videos_per_prompt
        self.cfg_scale, self.pag_scale = cfg_scale, pag_scale
        self.eval_generator = eval_generator

    def _norm(self, metric, key):
        range = self.METRICS_NORMALIZATION_RANGES[key] if key in self.METRICS_NORMALIZATION_RANGES else [0.0, 1.0]
        metric = max(metric, range[0])
        metric = min(metric, range[1])
        metric = (metric - range[0]) / (range[1] - range[0])
        return metric

    def evaluate(self, step, f_log):
        device = self.device

        eval_info_dict = {}

        from vbench.utils import init_submodules

        for category_id in self.cfg.category_ids:
            additional_cfg = {
                "batch_size": self.batch_size,
                "category_id": category_id,
                "shuffle": True,
                "num_samples": self.num_samples,
                "num_videos_per_prompt": self.num_videos_per_prompt,
            }

            if self.cfg.eval_data_providers[0] == "VBenchTextPrompt":
                cfg = VBenchTextPromptDataProviderConfig(**additional_cfg)
                data_provider = VBenchTextPromptDataProvider(cfg)
            elif self.cfg.eval_data_providers[0] == "VBenchImagePrompt":
                cfg = VBenchImagePromptDataProviderConfig(**additional_cfg)
                data_provider = VBenchImagePromptDataProvider(cfg)
            else:
                raise NotImplementedError(f"Vbench task {self.cfg.eval_data_providers[0]} is not supported")

            data_loader = data_provider.data_loader
            dimension_list = data_provider.dimension_list

            if self.cfg.resolution in ["480", "480F32", "480F64"]:
                resolution_ratio = "26-15"
            elif self.cfg.resolution in ["720F32", "720F64"]:
                resolution_ratio = "20-11"
            elif self.cfg.resolution == "720":
                resolution_ratio = "16-9"
            elif self.cfg.resolution in ["1080F32", "1080F64", "2160F32", "2160F64"]:
                resolution_ratio = "30-17"
            else:
                raise NotImplementedError(f"Unsupported resolution {self.cfg.resolution}")

            submodules_dict = init_submodules(dimension_list, local=True, read_frame=False, resolution=resolution_ratio)
            computer_list, key_list = build_metric_computer(dimension_list, device, submodules_dict)

            with tqdm(
                total=len(data_loader),
                desc="Valid Step #{}".format(step),
                disable=not is_master(),
                file=f_log,
                mininterval=10.0,
                bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            ) as t:
                for _, samples in enumerate(data_loader):
                    prompts = samples["prompt"]
                    extended_prompts = samples["extended_prompt"]
                    auxiliary_info = samples["auxiliary_info"] if "auxiliary_info" in samples else None

                    if self.cfg.model == "wan_i2v":
                        image_path_list = samples["image_path"]
                        image_list = [Image.open(image_path) for image_path in image_path_list]
                    elif self.cfg.model == "wan_t2v":
                        image_list = None
                    else:
                        raise ValueError(f"model {self.cfg.model} is not supported")

                    video_samples = self.trainer.generate(
                        self.network, extended_prompts, image_list, self.cfg_scale, self.pag_scale, self.eval_generator
                    )

                    if isinstance(video_samples, list):
                        video_samples = torch.stack(video_samples, dim=0)
                    video_samples_uint8 = torch.clamp(127.5 * video_samples + 127.5, 0, 255).to(dtype=torch.uint8)
                    video_samples_tensor = video_samples_uint8.permute(0, 2, 1, 3, 4).cpu().squeeze(0)

                    for computer, key in zip(computer_list, key_list):
                        if key in ["overall_consistency", "temporal_style", "human_action", "camera_motion"]:
                            computer.update_single(video_samples_tensor, prompts)
                        elif key in [
                            "object_class",
                            "multiple_objects",
                            "spatial_relationship",
                            "scene",
                            "appearance_style",
                        ]:
                            computer.update_single(video_samples_tensor, auxiliary_info)
                        elif key in ["color"]:
                            computer.update_single(video_samples_tensor, prompts, auxiliary_info)
                        elif key in ["i2v_subject", "i2v_background"]:
                            image_path = samples["image_path"][0]
                            computer.update_single(video_samples_tensor, image_path)
                        else:
                            computer.update_single(video_samples_tensor)

                    t.update()

                    for computer, key in zip(computer_list, key_list):
                        result = computer.compute()
                        if is_master():
                            print(key, result)

            torch.cuda.empty_cache()

            for computer, key in zip(computer_list, key_list):
                result = computer.compute()
                eval_info_dict[key] = result
                if is_master():
                    print(f"{key}: {result}")

        i2v_quality_keys = [
            "subject_consistency",
            "background_consistency",
            "motion_smoothness",
            "dynamic_degree",
            "aesthetic_quality",
            "imaging_quality",
        ]
        quality_keys = i2v_quality_keys + ["temporal_flickering"]
        semantic_keys = [
            "object_class",
            "multiple_objects",
            "human_action",
            "color",
            "spatial_relationship",
            "scene",
            "appearance_style",
            "temporal_style",
            "overall_consistency",
        ]
        i2v_keys = ["i2v_subject", "i2v_background", "camera_motion"]

        if all(key in eval_info_dict for key in i2v_quality_keys):
            eval_info_dict["i2v_quality_score"] = 0.0
            for key in quality_keys:
                result = self._norm(eval_info_dict[key], key)
                eval_info_dict["i2v_quality_score"] += result if key != "dynamic_degree" else result * 0.5
            eval_info_dict["i2v_quality_score"] = eval_info_dict["i2v_quality_score"] / 5.5

        if all(key in eval_info_dict for key in quality_keys):
            eval_info_dict["quality_score"] = 0.0
            for key in quality_keys:
                result = self._norm(eval_info_dict[key], key)
                eval_info_dict["quality_score"] += result if key != "dynamic_degree" else result * 0.5
            eval_info_dict["quality_score"] = eval_info_dict["quality_score"] / 6.5

        if all(key in eval_info_dict for key in semantic_keys):
            eval_info_dict["semantic_score"] = 0.0
            for key in semantic_keys:
                result = self._norm(eval_info_dict[key], key)
                eval_info_dict["semantic_score"] += result
            eval_info_dict["semantic_score"] = eval_info_dict["semantic_score"] / 9.0

        if all(key in eval_info_dict for key in i2v_keys):
            eval_info_dict["i2v_score"] = 0.0
            for key in quality_keys:
                result = self._norm(eval_info_dict[key], key)
                eval_info_dict["i2v_score"] += result if key != "camera_motion" else result * 0.1
            eval_info_dict["i2v_score"] = eval_info_dict["i2v_score"] / 2.1

        if "quality_score" in eval_info_dict and "semantic_score" in eval_info_dict:
            eval_info_dict["overall_score"] = (
                0.2 * eval_info_dict["semantic_score"] + 0.8 * eval_info_dict["quality_score"]
            )
        if "i2v_quality_score" in eval_info_dict and "i2v_score" in eval_info_dict:
            eval_info_dict["i2v_overall_score"] = (
                0.5 * eval_info_dict["i2v_quality_score"] + 0.5 * eval_info_dict["i2v_score"]
            )

        return eval_info_dict
