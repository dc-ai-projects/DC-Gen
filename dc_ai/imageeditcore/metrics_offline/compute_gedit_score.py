# Modified from GEdit Bench introduced by Shiyu Liu et. al. See https://github.com/stepfun-ai/Step1X-Edit.
# The original implementation is licensed under the Apache License 2.0.

import json
import math
import os
import random
from dataclasses import dataclass, field

import torch
from PIL import Image
from tqdm import tqdm

from ...apps.utils.config import get_config
from ...apps.utils.dist import (
    dist_barrier,
    dist_init,
    gather_list,
    get_dist_local_rank,
    get_dist_size,
    is_dist_initialized,
    is_master,
    sync_tensor,
)
from ..data_provider.gedit import GEditDataProvider, GEditDataProviderConfig
from .viescore import VIEScore


@dataclass
class ComputeGEditScoreConfig:
    images_dir: str = "tmp_qwen_image_edit/0/cfg_5.0"
    languages: str = "all"  # en, cn
    save_dir: str = "tmp_qwen_image_edit"
    llm: str = "qwen25vl"
    max_threads: int = 1
    concat_input_output: bool = True

    # GEdit
    raw_image_dir: str = "assets/data/GEdit/images"
    json_path: str = "assets/data/GEdit/meta_data.json"
    resolution: str = "1024"
    image_ext: str = ".jpg"
    ratio: float = 1.0
    gedit: GEditDataProviderConfig = field(
        default_factory=lambda: GEditDataProviderConfig(
            batch_size=1,
            resolution="${..resolution}",
        )
    )


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    new_area = width * height
    return int(width), int(height), int(new_area)


def process_single_item(input_image, edited_image, meta, vie_score):
    instruction = meta["prompt"][0]
    key = meta["name"][0]
    instruction_language = meta["language"][0]

    try:
        pil_image = input_image.convert("RGB")
        pil_image_edited = edited_image.convert("RGB")
        source_img_width, source_img_height, _ = calculate_dimensions(512 * 512, pil_image.width / pil_image.height)
        edited_img_width, edited_img_height, _ = calculate_dimensions(
            512 * 512, pil_image_edited.width / pil_image_edited.height
        )
        pil_image = pil_image.resize((int(source_img_width), int(source_img_height)))
        pil_image_edited = pil_image_edited.resize((int(edited_img_width), int(edited_img_height)))
        text_prompt = instruction

        if is_master():
            print("Begin Evaluation...")
        score_list = vie_score.evaluate([pil_image, pil_image_edited], text_prompt)
        sementics_score, quality_score, overall_score = score_list

        if is_master():
            print(
                f"sementics_score: {sementics_score}, quality_score: {quality_score}, overall_score: {overall_score}, instruction_language: {instruction_language}, instruction: {instruction}"
            )

        return {
            "key": key,
            "instruction": instruction,
            "sementics_score": sementics_score,
            "quality_score": quality_score,
            "overall_score": overall_score,
            "instruction_language": instruction_language,
            "success": True,
        }

    except Exception as e:
        return {
            "key": key,
            "instruction": instruction,
            "sementics_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "instruction_language": instruction_language,
            "success": False,
        }


def main():
    dist_init()
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())
    cfg: ComputeGEditScoreConfig = get_config(ComputeGEditScoreConfig)

    edited_images_dir = cfg.images_dir
    instruction_language = cfg.languages
    save_dir = cfg.save_dir
    backbone = cfg.llm
    groups = [
        "background_change",
        "color_alter",
        "material_alter",
        "motion_change",
        "ps_human",
        "style_change",
        "subject-add",
        "subject-remove",
        "subject-replace",
        "text_change",
        "tone_transfer",
    ]

    rank = get_dist_local_rank()
    world_size = get_dist_size()
    device = f"cuda:{rank}"
    for i in range(world_size):
        if rank == i:
            print(f"Rank {rank} is loading model on {device}...")
            vie_score = VIEScore(backbone=backbone, task="tie", device=device)
        dist_barrier()

    random.seed(42)

    save_dir_new = os.path.join(save_dir, backbone)
    os.makedirs(save_dir_new, exist_ok=True)
    all_meta = {}
    detailed_results_single_rank = {}

    for group_name in groups:
        cur_cfg = cfg.gedit
        cur_cfg.task_type = group_name
        data_provider = GEditDataProvider(cur_cfg)
        data_loader = data_provider.data_loader

        detailed_results_single_rank[group_name] = {
            "total_score": 0.0,
            "n_samples": 0,
            "mean_score": 0.0,
            "detailed_info": [],
        }
        total_score, n_samples = 0.0, 0

        if cfg.llm == "qwen25vl":
            for item in tqdm(data_loader, desc=f"Processing {group_name}"):
                key = item["name"][0]
                try:
                    img_PIL = Image.open(os.path.join(edited_images_dir, key + cfg.image_ext))
                except:
                    continue

                if cfg.concat_input_output:
                    height, width = img_PIL.height, img_PIL.width
                    input_img_PIL = img_PIL.crop((0, 0, width // 2, height))
                else:
                    input_img_PIL = Image.open(os.path.join(cfg.raw_image_dir, key + ".png"))

                if cfg.concat_input_output:
                    height, width = img_PIL.height, img_PIL.width
                    edited_img_PIL = img_PIL.crop((width // 2, 0, width, height))
                else:
                    edited_img_PIL = img_PIL

                result = process_single_item(input_img_PIL, edited_img_PIL, item, vie_score)
                detailed_results_single_rank[group_name]["detailed_info"].append(
                    {
                        "name": result["key"],
                        "language": result["instruction_language"],
                        "prompt": result["instruction"],
                        "sementics_score": str(result["sementics_score"]),
                        "quality_score": str(result["quality_score"]),
                        "overall_score": str(result["overall_score"]),
                    }
                )
                total_score += result["overall_score"]
                n_samples += 1
        else:
            raise ValueError(f"{cfg.llm} is not supported")

        if is_dist_initialized():
            total_score = torch.tensor(total_score).to(device)
            n_samples = torch.tensor(n_samples).to(device)
            total_score = sync_tensor(total_score, reduce="sum").cpu().numpy().item()
            n_samples = sync_tensor(n_samples, reduce="sum").cpu().numpy().item()
            local_info = detailed_results_single_rank[group_name]["detailed_info"]
            gathered_info = gather_list(local_info)

        if is_master():
            print(f"Mean Score of {group_name}: {total_score / n_samples}")
            all_meta[group_name] = {
                "total_score": total_score,
                "n_samples": n_samples,
                "mean_score": total_score / n_samples,
            }
            detailed_results_single_rank[group_name] = gathered_info
            print(detailed_results_single_rank[group_name])

        dist_barrier()

    if is_master():
        total_score, n_samples = 0.0, 0
        for key in all_meta.keys():
            if key in groups:
                total_score += all_meta[key]["total_score"]
                n_samples += all_meta[key]["n_samples"]
        all_meta["mean_score"] = total_score / n_samples

        print(all_meta)

        with open(os.path.join(save_dir_new, f"result_info_{cfg.languages}.json"), "w") as json_file:
            json.dump(all_meta, json_file, indent=4)
        with open(os.path.join(save_dir_new, f"detailed_result_info_{cfg.languages}.json"), "w") as json_file:
            json.dump(detailed_results_single_rank, json_file, indent=4)


if __name__ == "__main__":
    main()

"""
torchrun -m --nnodes=1 --nproc_per_node=8 dc_ai.imageeditcore.metrics_offline.compute_gedit_score \
    images_dir=exp/pico_banana/512to1024_eval_nrt/0/cfg_4.0 \
    save_dir=exp/pico_banana/512to1024_eval_nrt \
    resolution=1024 \
    llm=qwen25vl languages=all image_ext=.jpg 
"""
