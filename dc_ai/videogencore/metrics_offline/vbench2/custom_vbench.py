# VBench 2.0 was introduced by Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu in "VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness", see https://arxiv.org/abs/2503.21755.
# The original implementation of VBench 2.0 is by VBench Team, licensed under the Apache License 2.0. See https://github.com/Vchitect/VBench.
# This file is modified from https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/__init__.py, with the following changes:
# - return the results after evaluation
# - support resume evaluation
# - support evaluate multiple dimensions at once

import os

import decord
import vbench2.hack_registry  # to avoid error KeyError: 'Adafactor is already registered in optimizer at torch.optim'
from vbench2 import VBench2
from vbench2.utils import init_submodules, load_json, save_json

from ....apps.utils.dist import broadcast_object, dist_barrier, is_master


class CustomVBench2(VBench2):
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
        results_dict = {}
        if dimension_list is None:
            dimension_list = self.build_full_dimension_list()
        submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)

        if is_master():
            os.makedirs(self.output_path, exist_ok=True)
        output_path = os.path.join(self.output_path, "vbench2_eval_results.json")
        if os.path.exists(output_path):
            results_dict = load_json(output_path)
        dist_barrier()

        for dimension in dimension_list:
            if dimension in results_dict:
                if is_master():
                    print(f"Dimension {dimension} already evaluated, skipping")
                continue

            if is_master():
                cur_full_info_path = self.build_full_info_json(
                    os.path.join(videos_path, dimension), name, [dimension], prompt_list, mode=mode, **kwargs
                )
            else:
                cur_full_info_path = None
            cur_full_info_path = broadcast_object(cur_full_info_path)

            if dimension == "Human_Anatomy":
                from .human_anatomy import compute_human_anatomy

                evaluate_func = compute_human_anatomy
            elif dimension == "Human_Clothes":
                from .human_clothes import compute_human_clothes

                evaluate_func = compute_human_clothes
            elif dimension == "Human_Identity":
                from .human_identity import compute_human_identity

                evaluate_func = compute_human_identity
            elif dimension == "Composition":
                from .composition import compute_composition

                evaluate_func = compute_composition
            elif dimension == "Diversity":
                from .diversity import compute_diversity

                evaluate_func = compute_diversity
            elif dimension == "Mechanics":
                from .mechanics import compute_mechanics

                evaluate_func = compute_mechanics
            elif dimension == "Material":
                from .material import compute_material

                evaluate_func = compute_material
            elif dimension == "Thermotics":
                from .thermotics import compute_thermotics

                evaluate_func = compute_thermotics
            elif dimension == "Multi-View_Consistency":
                from .multi_view_consistency import compute_multi_view_consistency

                evaluate_func = compute_multi_view_consistency
            elif dimension == "Dynamic_Spatial_Relationship":
                from .dynamic_spatial_relationship import compute_dynamic_spatial_relationship

                evaluate_func = compute_dynamic_spatial_relationship
            elif dimension == "Dynamic_Attribute":
                from .dynamic_attribute import compute_dynamic_attribute

                evaluate_func = compute_dynamic_attribute
            elif dimension == "Motion_Order_Understanding":
                from .motion_order_understanding import compute_motion_order_understanding

                evaluate_func = compute_motion_order_understanding
            elif dimension == "Human_Interaction":
                from .human_interaction import compute_human_interaction

                evaluate_func = compute_human_interaction
            elif dimension == "Complex_Landscape":
                from .complex_landscape import compute_complex_landscape

                evaluate_func = compute_complex_landscape
            elif dimension == "Complex_Plot":
                from .complex_plot import compute_complex_plot

                evaluate_func = compute_complex_plot
            elif dimension == "Camera_Motion":
                from .camera_motion import compute_camera_motion

                evaluate_func = compute_camera_motion
            elif dimension == "Motion_Rationality":
                from .motion_rationality import compute_motion_rationality

                evaluate_func = compute_motion_rationality
            elif dimension == "Instance_Preservation":
                from .instance_preservation import compute_instance_preservation

                evaluate_func = compute_instance_preservation
            else:
                raise ValueError(f"Dimension {dimension} not supported")
            submodules_list = submodules_dict[dimension]
            results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
            results_dict[dimension] = results
            if is_master():
                save_json(results_dict, output_path)
                os.remove(cur_full_info_path)

            decord.bridge.reset_bridge()  # some dimensions may set the bridge, while some other dimensions require it to be "native", so we need to reset it

        return results_dict
