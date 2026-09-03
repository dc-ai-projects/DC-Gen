from dataclasses import dataclass

from .base_train import VideoGenCoreLatentTrainDataProvider, VideoGenCoreLatentTrainDataProviderConfig


@dataclass
class LatentFusionXTrainDataProviderConfig(VideoGenCoreLatentTrainDataProviderConfig):
    name: str = "LatentFusionXTrain"
    wds_meta_filename: str = "fusionX.json"
