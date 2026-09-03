# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/__init__.py, with the following changes:
# - return the results after evaluation
# - support resume evaluation


import importlib
import os

from vbench2_beta_i2v import VBenchI2V
from vbench2_beta_i2v.utils import init_submodules, load_json, save_json

from ....apps.utils.dist import dist_barrier, is_master


class CustomVBenchI2V(VBenchI2V):
    def build_full_info_json(
        self,
        videos_path,
        name,
        dimension_list,
        prompt_list=[],
        special_str="",
        verbose=False,
        mode="vbench_standard",
        **kwargs,
    ):
        info = {}
        if mode == "cosmos":
            video_name_to_motion_type = {}
            cur_full_info_list = []  # to save the prompt and video path info for the current dimensions
            full_info_list = load_json(self.full_info_dir)
            for prompt_dict in full_info_list:
                # if the prompt belongs to any dimension we want to evaluate
                if set(dimension_list) & set(prompt_dict["dimension"]):
                    video_prefix = prompt_dict["video_prefix"]
                    video_path = os.path.join(videos_path, f"{video_prefix}.mp4")
                    assert os.path.exists(video_path), f"Video {video_path} not found"
                    prompt_dict["video_list"] = [video_path]
                    cur_full_info_list.append(prompt_dict)
                    if "camera_motion" in prompt_dict["dimension"]:
                        prompt = prompt_dict["prompt_en"]
                        if "tilts down" in prompt:
                            motion_type = "tilt_down"
                        elif "tilts upward" in prompt or "tilt upwards" in prompt:
                            motion_type = "tilt_up"
                        elif "pans left" in prompt:
                            motion_type = "pan_left"
                        elif "pans right" in prompt:
                            motion_type = "pan_right"
                        elif "zooms in" in prompt or "zooming in" in prompt:
                            motion_type = "zoom_in"
                        elif "zooms out" in prompt:
                            motion_type = "zoom_out"
                        elif "remains static" in prompt:
                            motion_type = "static"
                        else:
                            raise ValueError(f"Unknown motion type: {prompt}")
                        video_name_to_motion_type[f"{video_prefix}.mp4"] = motion_type

            cur_full_info_path = os.path.join(self.output_path, name + "_full_info.json")
            save_json(cur_full_info_list, cur_full_info_path)
            info["video_name_to_motion_type"] = video_name_to_motion_type
            return cur_full_info_path, info
        else:
            return (
                super().build_full_info_json(
                    videos_path, name, dimension_list, prompt_list, special_str, verbose, mode, **kwargs
                ),
                info,
            )

    def evaluate(
        self,
        videos_path,
        name,
        mode="vbench_standard",
        dimension_list=None,
        local=False,
        read_frame=False,
        custom_prompt=False,
        resolution="1-1",
        **kwargs,
    ):
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame, resolution=resolution)
        cur_full_info_path, info = self.build_full_info_json(
            videos_path, name, dimension_list, custom_prompt=custom_prompt, mode=mode
        )
        if "video_name_to_motion_type" in info:
            kwargs["video_name_to_motion_type"] = info["video_name_to_motion_type"]

        if is_master():
            os.makedirs(self.output_path, exist_ok=True)
        output_path = os.path.join(self.output_path, "i2v_eval_results.json")
        if os.path.exists(output_path):
            results_dict = load_json(output_path)
        else:
            results_dict = {}
        dist_barrier()
        for dimension in dimension_list:
            if dimension in results_dict:
                if is_master():
                    print(f"Dimension {dimension} already evaluated, skipping")
                continue
            try:
                if dimension == "camera_motion":
                    from .camera_motion import compute_camera_motion

                    evaluate_func = compute_camera_motion
                elif dimension == "i2v_subject":
                    from .i2v_subject import compute_i2v_subject

                    evaluate_func = compute_i2v_subject
                elif dimension == "i2v_background":
                    from .i2v_background import compute_i2v_background

                    evaluate_func = compute_i2v_background
                else:
                    dimension_module = importlib.import_module(f"vbench.{dimension}")
                    evaluate_func = getattr(dimension_module, f"compute_{dimension}")
            except Exception as e:
                raise NotImplementedError(f"UnImplemented dimension {dimension}!, {e}")
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
            if is_master():
                save_json(results_dict, output_path)
        dist_barrier()
        if is_master():
            os.remove(cur_full_info_path)

        return results_dict
