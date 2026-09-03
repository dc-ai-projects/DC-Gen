# VBench was introduced by Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu in "VBench: Comprehensive Benchmark Suite for Video Generative Models", see https://arxiv.org/abs/2311.17982.
# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench and VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/vbench/__init__.py, with the following changes:
# - return the results after evaluation
# - support resume evaluation


import importlib
import os

from vbench import VBench
from vbench.utils import init_submodules, load_json, save_json

from ....apps.utils.dist import dist_barrier, is_master


class CustomVBench(VBench):
    def evaluate(
        self,
        videos_path,
        name,
        prompt_list=[],
        dimension_list=None,
        local=False,
        read_frame=False,
        mode="vbench_standard",
        **kwargs,
    ):
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        cur_full_info_path = self.build_full_info_json(
            videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs
        )

        if is_master():
            os.makedirs(self.output_path, exist_ok=True)
        output_path = os.path.join(self.output_path, "t2v_eval_results.json")
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
            dimension_module = importlib.import_module(f"vbench.{dimension}")
            evaluate_func = getattr(dimension_module, f"compute_{dimension}")
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
            if is_master():
                save_json(results_dict, output_path)
        dist_barrier()
        if is_master():
            os.remove(cur_full_info_path)

        return results_dict
