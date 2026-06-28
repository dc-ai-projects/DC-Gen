# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved. Licensed under the Apache License 2.0.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .attention import flash_attention
from .xlm_roberta import XLMRoberta

__all__ = [
    "XLMRobertaCLIP",
    "clip_xlm_roberta_vit_h_14",
    "CLIPModel",
]


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, s, c, n, d = *x.size(), self.num_heads, self.head_dim

        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        x = flash_attention(q, k, v)
        x = x.reshape(b, s, c)

        x = self.proj(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, num_heads, norm_eps=1e-5):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.norm_eps = norm_eps

        self.norm1 = LayerNorm(dim, eps=norm_eps)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = LayerNorm(dim, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        dim=768,
        mlp_ratio=4,
        out_dim=512,
        num_heads=12,
        num_layers=12,
        norm_eps=1e-5,
    ):

        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.norm_eps = norm_eps

        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)

        self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(1, self.num_patches + 1, dim))

        self.pre_norm = LayerNorm(dim, eps=norm_eps)
        self.transformer = nn.Sequential(
            *[AttentionBlock(dim, mlp_ratio, num_heads, norm_eps) for _ in range(num_layers)]
        )
        self.post_norm = LayerNorm(dim, eps=norm_eps)

        self.head = nn.Parameter(gain * torch.randn(dim, out_dim))

    def forward(self, x):
        b = x.size(0)

        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        x = self.pre_norm(x)

        x = self.transformer[:-1](x)
        return x


class XLMRobertaWithHead(XLMRoberta):
    def __init__(self, **kwargs):
        self.out_dim = kwargs.pop("out_dim")
        super().__init__(**kwargs)

        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.Sequential(
            nn.Linear(self.dim, mid_dim, bias=False),
            nn.GELU(),
            nn.Linear(mid_dim, self.out_dim, bias=False),
        )

    def forward(self, ids):
        x = super().forward(ids)

        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Module):
    def __init__(
        self,
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.norm_eps = norm_eps

        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            norm_eps=norm_eps,
        )
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
        )
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))

    def forward(self, imgs, txt_ids):
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt


def _clip(model_cls=XLMRobertaCLIP, dtype=torch.float32, device="cpu", **kwargs):
    with torch.device(device):
        model = model_cls(**kwargs)

    model = model.to(dtype=dtype, device=device)
    output = (model,)

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    transforms = T.Compose(
        [
            T.Resize((model.image_size, model.image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    output += (transforms,)

    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(dtype, device):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        dtype=dtype,
        device=device,
    )
    return _clip(XLMRobertaCLIP, **cfg)


class CLIPModel:
    def __init__(self, dtype, device, checkpoint_path):
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path

        self.model, self.transforms = clip_xlm_roberta_vit_h_14(dtype=dtype, device=device)
        self.model = self.model.eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    def visual(self, videos):
        size = (self.model.image_size,) * 2
        videos = torch.cat(
            [F.interpolate(u.transpose(0, 1), size=size, mode="bicubic", align_corners=False) for u in videos]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos)
            return out
