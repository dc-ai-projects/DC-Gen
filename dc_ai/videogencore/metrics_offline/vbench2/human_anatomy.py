# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/human_anatomy.py, with the following changes:
# - support distributed evaluation

import os

import vbench2
from vbench2.third_party.ViTDetector.detect import compute_abnormality
from vbench2.utils import load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master


def compute_human_anatomy(json_dir, device, submodules_dict, **kwargs):
    # config_base_path = vbench2.__path__[0].removesuffix("vbench2")
    # submodules_dict["detector_config"] = os.path.join(config_base_path, submodules_dict["detector_config"])
    # for key in ["human", "face", "hand"]:
    #     submodules_dict["analyzer_configs"][key]["cfg_path"] = os.path.join(config_base_path, submodules_dict["analyzer_configs"][key]["cfg_path"])

    video_list, _ = load_dimension_info(json_dir, dimension="human_anatomy", lang="en")
    video_list = distribute_list_to_rank(video_list)
    video_list = [os.path.abspath(video) for video in video_list]

    original_dir = os.getcwd()
    os.chdir(
        vbench2.__path__[0].removesuffix("vbench2")
    )  # Too many relative paths in vbench2, so we change directory to the base path
    if is_master():
        print(f"Now in: {os.getcwd()}")

    all_results, video_results = compute_abnormality(video_list, device, submodules_dict, **kwargs)

    os.chdir(original_dir)
    if is_master():
        print(f"Returned to: {os.getcwd()}")

    if get_dist_size() > 1:
        video_results = gather_list(video_results)
    if is_master():
        all_results = sum([x["video_results"] for x in video_results]) / len(video_results)
        return all_results, video_results
    else:
        return None, None
