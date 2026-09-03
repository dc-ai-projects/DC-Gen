# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/human_identity.py, with the following changes:
# - support distributed evaluation

import decord
import torch
from retinaface.predict_single import Model
from torch.utils import model_zoo
from vbench2.human_identity import evaluate_id_consistency
from vbench2.third_party.arcface.models import resnet_face18
from vbench2.utils import load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master


def compute_human_identity(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="human_identity", lang="en")

    url = "https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip"
    retina_state_dict = model_zoo.load_url(url, progress=True, map_location="cpu")
    retina_model = Model(max_size=2048, device=device)
    retina_model.load_state_dict(retina_state_dict)
    model = resnet_face18(use_se=False)
    state_dict = torch.load(submodules_dict["model"])
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.to(device).eval()

    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    _, video_results = evaluate_id_consistency(prompt_dict_ls, retina_model, model)
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
