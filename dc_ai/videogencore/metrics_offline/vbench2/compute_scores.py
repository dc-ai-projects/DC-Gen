# please run this script in the vbench2 image, see projects/DC-VideoGen/VBench2.md for more details

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import torch
from omegaconf import MISSING

from ....apps.utils.config import get_config
from ....apps.utils.dist import dist_init, get_dist_local_rank, is_dist_initialized, is_master


@dataclass
class VBench2Config:
    full_info_path: str = "assets/data/vbench/VBench2_full_info.json"
    videos_path: str = MISSING
    output_dir: str = "${.videos_path}"
    dimension_list: Optional[tuple[str, ...]] = None


class VBench2Wrapper:
    def __init__(self, cfg: VBench2Config, device: torch.device):
        self.cfg = cfg
        self.device = device

    def evaluate(self):
        from .custom_vbench import CustomVBench2

        vbench = CustomVBench2(self.device, self.cfg.full_info_path, self.cfg.output_dir)

        results = vbench.evaluate(
            videos_path=self.cfg.videos_path,
            name="placeholder",
            dimension_list=self.cfg.dimension_list,
        )
        results = {key: value[0] for key, value in results.items()}
        if is_master() and self.cfg.dimension_list is None:
            results["Creativity_Score"] = np.mean([results["Diversity"], results["Composition"]]).item()
            results["Commonsense_Score"] = np.mean(
                [results["Motion_Rationality"], results["Instance_Preservation"]]
            ).item()
            results["Controllability_Score"] = np.mean(
                [
                    results["Dynamic_Spatial_Relationship"],
                    results["Dynamic_Attribute"],
                    results["Motion_Order_Understanding"],
                    results["Human_Interaction"],
                    results["Complex_Landscape"],
                    results["Complex_Plot"],
                    results["Camera_Motion"],
                ]
            ).item()
            results["Human_Fidelity_Score"] = np.mean(
                [results["Human_Anatomy"], results["Human_Identity"], results["Human_Clothes"]]
            ).item()
            results["Physics_Score"] = np.mean(
                [results["Mechanics"], results["Thermotics"], results["Material"], results["Multi-View_Consistency"]]
            ).item()
            results["Total_Score"] = np.mean(
                [
                    results["Creativity_Score"],
                    results["Commonsense_Score"],
                    results["Controllability_Score"],
                    results["Human_Fidelity_Score"],
                    results["Physics_Score"],
                ]
            ).item()
        return results


def main():
    dist_init(timeout=timedelta(minutes=60))
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())
    device = torch.device("cuda")
    cfg = get_config(VBench2Config)
    vbench2 = VBench2Wrapper(cfg, device)
    results = vbench2.evaluate()
    if is_master():
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
