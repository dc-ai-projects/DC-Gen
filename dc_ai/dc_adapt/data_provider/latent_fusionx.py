from dataclasses import dataclass

from torch.utils.data import Dataset

from .base import DCAdaptAlignVideoLatentDataset, DCAdaptLatentDataProvider, DCAdaptLatentDataProviderConfig

__all__ = [
    "DCAdaptLatentFusionXDataProviderConfig",
    "DCAdaptLatentFusionXDataProvider",
]


@dataclass
class DCAdaptLatentFusionXDataProviderConfig(DCAdaptLatentDataProviderConfig):
    name: str = "LatentFusionXAlign"

    teacher_wds_meta_filename: str = "fusionX_align.json"
    student_wds_meta_filename: str = "fusionX_align.json"


class DCAdaptLatentFusionXDataProvider(DCAdaptLatentDataProvider):
    def __init__(self, cfg: DCAdaptLatentFusionXDataProviderConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptLatentFusionXDataProviderConfig

    def build_dataset(self) -> Dataset:
        dataset = DCAdaptAlignVideoLatentDataset(cfg=self.cfg)
        return dataset
