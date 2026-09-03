# please run this script in the vbench environment

from dataclasses import dataclass
from typing import Optional

import torch
from omegaconf import MISSING

from ....apps.utils.config import get_config
from ....apps.utils.dist import dist_barrier


@dataclass
class VBenchT2VConfig:
    full_info_path: str = "assets/data/vbench/VBench_full_info.json"
    videos_path: str = MISSING
    output_dir: str = "${.videos_path}"
    dimension_list: Optional[tuple[str, ...]] = None


class VBenchT2VWrapper:
    def __init__(self, cfg: VBenchT2VConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

    def evaluate(self):
        from .constants import DIM_WEIGHT, NORMALIZE_DIC, QUALITY_LIST, QUALITY_WEIGHT, SEMANTIC_LIST, SEMANTIC_WEIGHT
        from .custom_vbench import CustomVBench

        vbench = CustomVBench(self.device, self.cfg.full_info_path, self.cfg.output_dir)

        results = vbench.evaluate(
            videos_path=self.cfg.videos_path,
            name="placeholder",
            dimension_list=self.cfg.dimension_list,
        )
        results = {key: value[0] for key, value in results.items()}
        dist_barrier()

        if self.cfg.dimension_list is None:
            normalized_score = {}
            for key in NORMALIZE_DIC:
                normalized_score[key] = (
                    (results[key] - NORMALIZE_DIC[key]["Min"])
                    / (NORMALIZE_DIC[key]["Max"] - NORMALIZE_DIC[key]["Min"])
                    * DIM_WEIGHT[key]
                )
            results["quality_score"] = sum([normalized_score[key] for key in QUALITY_LIST]) / sum(
                [DIM_WEIGHT[key] for key in QUALITY_LIST]
            )
            results["semantic_score"] = sum([normalized_score[key] for key in SEMANTIC_LIST]) / sum(
                [DIM_WEIGHT[key] for key in SEMANTIC_LIST]
            )
            results["total_score"] = (
                results["quality_score"] * QUALITY_WEIGHT + results["semantic_score"] * SEMANTIC_WEIGHT
            ) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT)

        return results


def main():
    from ....apps.utils.dist import dist_init, get_dist_local_rank, is_dist_initialized, is_master

    dist_init()
    if is_dist_initialized():
        torch.cuda.set_device(get_dist_local_rank())
    device = torch.device("cuda")
    cfg = get_config(VBenchT2VConfig)
    vbench_t2v = VBenchT2VWrapper(cfg, device)
    results = vbench_t2v.evaluate()
    if is_master():
        for key, value in results.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
