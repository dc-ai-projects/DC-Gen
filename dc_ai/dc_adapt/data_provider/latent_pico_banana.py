from dataclasses import dataclass

from torch.utils.data import Dataset

from .base import DCAdaptAlignImageEditLatentDataset, DCAdaptLatentDataProvider, DCAdaptLatentDataProviderConfig

__all__ = [
    "DCAdaptLatentPicoBananaDataProviderConfig",
    "DCAdaptLatentPicoBananaDataProvider",
]


@dataclass
class DCAdaptLatentPicoBananaDataProviderConfig(DCAdaptLatentDataProviderConfig):
    name: str = "LatentPicoBananaAlign"
    resolution: str = "512F32MS"

    teacher_wds_meta_filename: str = "pico_banana_single.json"  # Single Turn Editing Data
    student_wds_meta_filename: str = "pico_banana_single.json"

    data_ext: str = ".pth"


@dataclass
class DCAdaptLatentPicoBananaQwenImageGenDataProviderConfig(DCAdaptLatentPicoBananaDataProviderConfig):
    name: str = "LatentPicoBananaQwenImageGenAlign"

    teacher_wds_meta_filename: str = "pico_banana_qwen_image_gen.json"
    student_wds_meta_filename: str = "pico_banana_qwen_image_gen.json"

    data_ext: str = ".pth"


class DCAdaptLatentPicoBananaDataProvider(DCAdaptLatentDataProvider):
    def __init__(self, cfg: DCAdaptLatentPicoBananaDataProviderConfig):
        super().__init__(cfg)
        self.cfg: DCAdaptLatentPicoBananaDataProviderConfig

    def build_dataset(self) -> Dataset:
        dataset = DCAdaptAlignImageEditLatentDataset(cfg=self.cfg)
        return dataset
