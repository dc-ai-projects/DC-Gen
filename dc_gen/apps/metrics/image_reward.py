# Image Reward was introduced by Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong in "ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation", see https://arxiv.org/abs/2304.05977.
# The original implementation is by Jiazheng Xu, Xiao Liu, licensed under the Apache License 2.0. See https://github.com/THUDM/ImageReward.

import os
from dataclasses import dataclass
from typing import Union

import torch
from ImageReward.ImageReward import ImageReward
from ImageReward.utils import _MODELS, ImageReward_download, available_models
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode, to_pil_image

from ..utils.dist import sync_tensor


def silent_load(
    name: str = "ImageReward-v1.0",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    download_root: str = None,
    med_config: str = None,
):
    """Load a ImageReward model

    Parameters
    ----------
    name : str
        A model name listed by `ImageReward.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    download_root: str
        path to download the model files; by default, it uses "~/.cache/ImageReward"

    Returns
    -------
    model : torch.nn.Module
        The ImageReward model
    """
    if name in _MODELS:
        model_path = ImageReward_download(_MODELS[name], download_root or os.path.expanduser("~/.cache/ImageReward"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # print('load checkpoint from %s'%model_path)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # med_config
    if med_config is None:
        med_config = ImageReward_download(
            "https://huggingface.co/THUDM/ImageReward/blob/main/med_config.json",
            download_root or os.path.expanduser("~/.cache/ImageReward"),
        )

    model = ImageReward(device=device, med_config=med_config).to(device)
    msg = model.load_state_dict(state_dict, strict=False)
    # print("checkpoint loaded")
    model.eval()

    return model


@dataclass
class ImageRewardStatsConfig:
    pass


class ImageRewardStats:
    def __init__(self, cfg: ImageRewardStatsConfig):
        self.cfg = cfg
        self.device = torch.device("cuda")
        self.model = silent_load("ImageReward-v1.0", self.device)
        self.sum_reward, self.cnt = 0, 0
        self.transform = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ]
        )

    @torch.no_grad()
    def add_data(self, batch: torch.Tensor, prompts: list[str]):
        # batch: (B, C, H, W)
        if isinstance(batch, torch.Tensor):
            if batch.dtype == torch.uint8:
                batch_uint8 = batch
            elif batch.dtype == torch.float32:
                # to simulate storing and loading generated images
                # reference: torchvision save_image
                # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
                batch_uint8 = (255 * batch + 0.5).clamp(0, 255).to(torch.uint8)
            else:
                raise NotImplementedError(f"dtype {batch.dtype} is not supported")
        else:
            raise TypeError(type(batch))

        # text encode
        text_input = self.model.blip.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
        ).to(self.device)

        batch = torch.stack([self.transform(to_pil_image(image_uint8)) for image_uint8 in batch_uint8], dim=0).to(
            self.device
        )

        # results using torch or torchvision are different
        # batch = torch.nn.functional.interpolate(batch.float(), size=(224, 224), mode='bicubic', antialias=True).clamp(0, 255)
        # # batch = resize(batch_uint8.float(), (224, 224), InterpolationMode.BICUBIC, None, True).round_().clamp(0, 255)
        # batch = batch / 255
        # batch = normalize(batch, (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711), True)

        batch_embeds = self.model.blip.visual_encoder(batch)

        # text encode cross attention with image
        image_atts = torch.ones(batch_embeds.size()[:-1], dtype=torch.long, device=self.device)
        text_output = self.model.blip.text_encoder(
            text_input.input_ids,
            attention_mask=text_input.attention_mask,
            encoder_hidden_states=batch_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        txt_features = text_output.last_hidden_state[:, 0, :].float()  # (feature_dim)
        rewards = self.model.mlp(txt_features)
        rewards = (rewards - self.model.mean) / self.model.std

        self.sum_reward += rewards.sum().item()
        self.cnt += rewards.numel()

    def compute(self):
        sum_reward = float(sync_tensor(self.sum_reward, reduce="sum"))
        cnt = int(sync_tensor(self.cnt, reduce="sum"))
        return sum_reward / cnt

    def reset(self):
        self.sum_reward, self.cnt = 0, 0
