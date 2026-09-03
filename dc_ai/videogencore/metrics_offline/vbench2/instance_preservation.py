# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/instance_preservation.py, with the following changes:
# - support distributed evaluation

import os
import subprocess

from swift.llm import (
    AdapterRequest,
    BaseArguments,
    InferRequest,
    PtEngine,
    RequestConfig,
    get_model_tokenizer,
    get_template,
)
from swift.tuners import Swift
from tqdm import tqdm
from vbench2.third_party.Instance_detector.split import get_video_info as get_video_info_split
from vbench2.third_party.Instance_detector.test import get_video_info, infer_lora
from vbench2.utils import load_dimension_info

from ....apps.utils.dist import distribute_list_to_rank, gather_list, get_dist_size, is_master


def split_video(filepath, output_folder):
    filename = os.path.basename(filepath)
    name, _ = os.path.splitext(filename)
    duration, fps = get_video_info_split(filepath)
    total_segments = int(duration) - 1

    for i in range(total_segments):
        out_path = os.path.join(output_folder, f"{name}_clip_{i:04d}.mp4")
        if os.path.exists(out_path):
            continue
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            filepath,
            "-ss",
            str(i),
            "-t",
            "2",
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            out_path,
        ]
        subprocess.run(cmd, stderr=subprocess.DEVNULL, check=True)


def compute_anomaly(prompt_dict_ls, device, submodules_dict, output_dir):
    processed_json = []
    request_config = RequestConfig(max_tokens=512, temperature=0)
    adapter_path = submodules_dict["model"]
    args = BaseArguments.from_pretrained(adapter_path)
    model, tokenizer = get_model_tokenizer(args.model, device_map=device)
    model = Swift.from_pretrained(model, adapter_path)
    template = get_template(args.template, tokenizer, args.system)
    engine = PtEngine.from_model_template(model, template)
    final_score = 0
    final_num = 0
    processed_json = []
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict["video_list"]
        for video_path in video_paths:
            split_video(video_path, output_dir)
            frame_count, fps, video_length = get_video_info(video_path)
            os.environ["FPS"] = f"{fps}"
            valid = True
            new_item = {
                "video_path": video_path,
            }
            for idx in range(int(video_length) - 1):
                clip_path = os.path.join(output_dir, f"{video_path.split('/')[-1][:-4]}_clip_{idx:04d}.mp4")
                message = [
                    {"role": "system", "content": "You are a helpful and harmless assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": clip_path},
                            {
                                "type": "text",
                                "text": "Does the video contain one or more of the following anomalies: sudden appearance, disappearance, fusion, fission?\nOptions:\nA. Yes\nB. No",
                            },
                        ],
                    },
                ]
                infer_request = InferRequest(messages=message)
                output_text = infer_lora(engine, request_config, infer_request)
                if "(A)" in output_text or "yes" in output_text.lower() or "A" == output_text:
                    valid = False
                    break
            for idx in range(int(video_length) - 1):
                clip_path = os.path.join(output_dir, f"{video_path.split('/')[-1][:-4]}_clip_{idx:04d}.mp4")
                os.remove(clip_path)
            if valid:
                final_score += 1
                new_item["video_results"] = 1.0
            else:
                new_item["video_results"] = 0.0
            final_num += 1
            processed_json.append(new_item)
    return final_score / final_num, processed_json


def compute_instance_preservation(json_dir, device, submodules_dict, **kwargs):
    # To deal with the key missing issue in VBench/VBench-2.0/vbench2/third_party/Instance_detector/swift/llm/model/model/qwen.py:564
    from qwen_vl_utils import vision_process

    vision_process.IMAGE_FACTOR = None
    vision_process.MIN_PIXELS = None
    vision_process.MAX_PIXELS = None
    vision_process.VIDEO_MIN_PIXELS = None
    vision_process.VIDEO_MAX_PIXELS = None
    vision_process.VIDEO_TOTAL_PIXELS = None

    _, prompt_dict_ls = load_dimension_info(json_dir, dimension="instance_preservation", lang="en")
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    _, video_results = compute_anomaly(prompt_dict_ls, device, submodules_dict, output_dir=os.path.dirname(json_dir))

    if get_dist_size() > 1:
        video_results = gather_list(video_results)
    if is_master():
        all_results = sum([d["video_results"] for d in video_results]) / len(video_results)
        return all_results, video_results
    else:
        return None, None
