# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/i2v_subject.py, with the following changes:
# - support distributed evaluation


import torch
from vbench2_beta_i2v.i2v_subject import i2v_subject, logger
from vbench.distributed import distribute_list_to_rank, gather_list_of_dict, get_world_size

from .utils import load_i2v_dimension_info


def compute_i2v_subject(json_dir, device, submodules_list, **kwargs):
    dino_model = torch.hub.load(**submodules_list).to(device)
    resolution = submodules_list["resolution"]
    logger.info("Initialize DINO success")
    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension="i2v_subject", lang="en", resolution=resolution)
    video_pair_list = distribute_list_to_rank(video_pair_list)
    all_results, video_results = i2v_subject(dino_model, video_pair_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
    return all_results, video_results
