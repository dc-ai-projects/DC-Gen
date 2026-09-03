# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/multi_view_consistency.py, with the following changes:
# - support distributed evaluation

from easydict import EasyDict as edict
from tqdm import tqdm
from vbench2.multi_view_consistency import CameraPredict, DynamicDegree, dynamic_degree, whether_orbit
from vbench2.utils import load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master


def multi_view_consistency(prompt_dict_ls, camera, dynamic):
    final_score = 0
    valid_num = 0
    processed_json = []
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict["video_list"]
        for video_path in video_paths:
            judge, skip_frame, end_frame = whether_orbit(video_path, camera)
            score = dynamic_degree(dynamic, video_path, skip_frame, end_frame, judge)
            processed_json.append({"video_path": video_path, "video_results": score})
            if score != -1:
                final_score += score
                valid_num += 1
    return None, processed_json


def compute_multi_view_consistency(json_dir, device, submodules_dict, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    model_path = submodules_dict["raft"]
    args_new = edict({"model": model_path, "small": False, "mixed_precision": False, "alternate_corr": False})
    dynamic = DynamicDegree(args_new, device)

    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="multi-view_consistency", lang="en")
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    _, video_results = multi_view_consistency(prompt_dict_ls, camera, dynamic)
    if get_dist_size() > 1:
        video_results = gather_list(video_results)
    if is_master():
        score = 0
        num = 0
        for d in video_results:
            if d["video_results"] != -1:
                num += 1
                score += d["video_results"]
        all_results = score / num
        return all_results, video_results
    else:
        return None, None
