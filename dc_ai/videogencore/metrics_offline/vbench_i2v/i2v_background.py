# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/i2v_background.py, with the following changes:
# - support distributed evaluation
# - change the cache_dir of dreamsim to "~/.cache/vbench/dreamsim"


import os

from dreamsim import dreamsim
from vbench2_beta_i2v.i2v_background import i2v_background, logger
from vbench.distributed import distribute_list_to_rank, gather_list_of_dict, get_world_size

from ....apps.utils.dist import dist_barrier, is_master
from .utils import load_i2v_dimension_info


def compute_i2v_background(json_dir, device, submodules_list, **kwargs):
    cache_dir = os.path.expanduser("~/.cache/vbench/dreamsim")
    if is_master():  # only download the model on rank 0
        dream_model, preprocess = dreamsim(pretrained=True, cache_dir=cache_dir)
    dist_barrier()
    if not is_master():
        dream_model, preprocess = dreamsim(pretrained=True, cache_dir=cache_dir)
    resolution = submodules_list["resolution"]
    logger.info("Initialize DreamSim success")

    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension="i2v_background", lang="en", resolution=resolution)
    video_pair_list = distribute_list_to_rank(video_pair_list)
    all_results, video_results = i2v_background(dream_model, video_pair_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
    return all_results, video_results
