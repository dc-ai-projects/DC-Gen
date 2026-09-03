# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/diversity.py, with the following changes:
# - support distributed evaluation
# - run evaluation on gpu

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from vbench2.diversity import VGG, content_loss, style_loss
from vbench2.utils import get_frames, load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master


def evaluate(style_features, content_features, device):
    content_diversity = 0
    style_diversity = 0
    len_seed = len(content_features)
    for i in range(len_seed):
        for j in range(i + 1, len_seed):
            content_diversity += content_loss(content_features[i].to(device), content_features[j].to(device)).item()
            for k in range(5):
                style_diversity += style_loss(style_features[i][k].to(device), style_features[j][k].to(device)).item()
    content_diversity /= 0.5 * len_seed * (len_seed - 1)
    style_diversity /= 2.5 * len_seed * (len_seed - 1)
    diversity = (content_diversity + 1000 * style_diversity) / 2
    return content_diversity, 1000 * style_diversity, diversity / 17.712  # Empirical maximum


def Diversity(prompt_dict_ls, model, device):
    final_score = 0
    processed_json = []
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict["video_list"]
        style_features = []
        content_features = []
        for video_path in video_paths:
            frames = get_frames(video_path)
            frames = torch.cat(frames, dim=0)
            frames = frames.to(device)
            with torch.no_grad():
                features = model(frames)
            style = features[:5]
            content = features[5]
            style_features.append(style)
            content_features.append(content)
            del style, content, frames
            torch.cuda.empty_cache()

        content_diversity, style_diversity, diversity = evaluate(style_features, content_features, device)
        diversity = np.clip(diversity, a_min=0, a_max=1)
        new_item = {"video_path": video_paths[0], "video_results": diversity.tolist()}
        processed_json.append(new_item)
        final_score += diversity
    return final_score / len(prompt_dict_ls), processed_json


def compute_diversity(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="diversity", lang="en")
    model = VGG().to(device)

    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    _, video_results = Diversity(prompt_dict_ls, model, device)
    if get_dist_size() > 1:
        video_results = gather_list(video_results)
    if is_master():
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
        return all_results, video_results
    else:
        return None, None
