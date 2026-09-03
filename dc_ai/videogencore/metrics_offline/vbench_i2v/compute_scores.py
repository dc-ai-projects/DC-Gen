# please run this script in the vbench environment

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
from omegaconf import MISSING

from ....apps.utils.config import get_config
from ....apps.utils.dist import dist_barrier


@dataclass
class VBenchI2VConfig:
    full_info_path: str = "assets/data/vbench/vbench2_i2v_full_info.json"
    videos_path: str = MISSING
    output_dir: str = "${.videos_path}"
    dimension_list: Optional[tuple[str, ...]] = None
    resolution: str = "1-1"
    mode: str = "vbench_standard"


class VBenchI2VWrapper:
    def __init__(self, cfg: VBenchI2VConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def evaluate(self):
        from .constants import (
            DIM_WEIGHT_I2V,
            I2V_LIST,
            I2V_QUALITY_LIST,
            I2V_QUALITY_WEIGHT,
            I2V_WEIGHT,
            NORMALIZE_DIC_I2V,
        )
        from .custom_vbench import CustomVBenchI2V

        vbench = CustomVBenchI2V(self.device, self.cfg.full_info_path, self.cfg.output_dir)

        results = vbench.evaluate(
            videos_path=self.cfg.videos_path,
            name="placeholder",
            mode=self.cfg.mode,
            dimension_list=self.cfg.dimension_list,
            resolution=self.cfg.resolution,
        )
        results = {key: value[0] for key, value in results.items()}
        dist_barrier()

        if self.cfg.dimension_list is None:
            normalized_score = {}
            for key in NORMALIZE_DIC_I2V:
                normalized_score[key] = (
                    (results[key] - NORMALIZE_DIC_I2V[key]["Min"])
                    / (NORMALIZE_DIC_I2V[key]["Max"] - NORMALIZE_DIC_I2V[key]["Min"])
                    * DIM_WEIGHT_I2V[key]
                )
            results["quality_score"] = sum([normalized_score[key] for key in I2V_QUALITY_LIST]) / sum(
                [DIM_WEIGHT_I2V[key] for key in I2V_QUALITY_LIST]
            )
            results["i2v_score"] = sum([normalized_score[key] for key in I2V_LIST]) / sum(
                [DIM_WEIGHT_I2V[key] for key in I2V_LIST]
            )
            results["total_score"] = (
                results["quality_score"] * I2V_QUALITY_WEIGHT + results["i2v_score"] * I2V_WEIGHT
            ) / (I2V_QUALITY_WEIGHT + I2V_WEIGHT)

        return results


def main():
    from ....apps.utils.dist import dist_init, get_dist_local_rank, is_dist_initialized, is_master

    dist_init(timeout=timedelta(minutes=60))
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())
    device = torch.device("cuda")
    cfg = get_config(VBenchI2VConfig)
    vbench_i2v = VBenchI2VWrapper(cfg, device)
    results = vbench_i2v.evaluate()
    if is_master():
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
