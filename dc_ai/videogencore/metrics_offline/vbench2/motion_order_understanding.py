# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/motion_order_understanding.py, with the following changes:
# - support distributed evaluation

from transformers import AutoModelForCausalLM, AutoTokenizer
from vbench2.motion_order_understanding import LLaVA_Video
from vbench2.utils import load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master
from .builder import load_pretrained_model


def compute_motion_order_understanding(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="motion_order_understanding", lang="en")
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    model_name = "llava_qwen"

    pretrained = submodules_dict["llava"]
    llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, torch_dtype="bfloat16", device_map=device
    )  # Add any other thing you want to pass in llava_model_args
    llava_model.eval()

    qwen_model_name = submodules_dict["qwen"]
    qwen_model = AutoModelForCausalLM.from_pretrained(
        qwen_model_name, torch_dtype="auto", device_map=device, cache_dir=submodules_dict["qwen"]
    )
    qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, cache_dir=submodules_dict["qwen"])

    _, video_results = LLaVA_Video(
        prompt_dict_ls, llava_model, llava_tokenizer, image_processor, qwen_model, qwen_tokenizer, device
    )
    if get_dist_size() > 1:
        video_results = gather_list(video_results)
    if is_master():
        score = 0
        num = 0
        count = 0
        for d in video_results:
            if d["video_results"] != -1:
                num += 1
                score += d["video_results"]
            else:
                count += 1
        if count > 0.9 * len(video_results):
            return 0, video_results
        all_results = score / num
        return all_results, video_results
    else:
        return None, None
