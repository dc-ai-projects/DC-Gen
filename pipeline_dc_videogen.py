"""
DC-VideoGen pipeline builders — T2V and I2V.
Source code is bundled under dc_videogen/ (no external repo paths needed).
Checkpoints are auto-downloaded from HuggingFace on first use.
"""

import os
import pathlib

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

# ── HuggingFace repo ──────────────────────────────────────────────────────────
HUB_REPO_VIDEOGEN = 'nvidia/DC-VideoGen-Wan2.1-14B'

# ── local cache (mirrors pretrained_models/ pattern used for image pipelines) ─
_repo_root = pathlib.Path(__file__).resolve().parent
CKPT = _repo_root / 'pretrained_models' / 'DC-Gen-Wan2.1-14B-720P'

_REQUIRED_CKPT_PATHS = [
    'dc-ae-v-f32t4c32-1.0-bf16.pt',
    'transformer_t2v/config.json',
    'transformer_i2v/config.json',
    'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
]

def _ensure_videogen_ckpt() -> pathlib.Path:
    """Download checkpoints from HF if any required file is missing."""
    if not all((CKPT / f).exists() for f in _REQUIRED_CKPT_PATHS):
        missing = [f for f in _REQUIRED_CKPT_PATHS if not (CKPT / f).exists()]
        print(f'[VideoGen] Missing checkpoints: {missing}')
        print(f'[VideoGen] Downloading from {HUB_REPO_VIDEOGEN} ...')
        CKPT.mkdir(parents=True, exist_ok=True)
        token = os.environ.get('HF_TOKEN')
        snapshot_download(
            repo_id=HUB_REPO_VIDEOGEN,
            repo_type='model',
            local_dir=str(CKPT),
            token=token,
        )
    return CKPT

from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from transformers import CLIPImageProcessor, T5TokenizerFast, UMT5EncoderModel

from dc_videogen.dc_ae_v import DCAEV, dc_ae_v_f32t4_chunk_causal
from dc_videogen.pipeline_dc_videogen_wan_t2v import DCVideoGenWanTextToVideoPipeline
from dc_videogen.pipeline_dc_videogen_wan_i2v import DCVideoGenWanImageToVideoPipeline


# ── VAE wrapper ───────────────────────────────────────────────────────────────

class AEWrapper(nn.Module):
    def __init__(self, model_name: str, model_path: str):
        super().__init__()
        self.config = type('C', (), {
            'scale_factor_temporal': 4,
            'scale_factor_spatial':  32,
            'z_dim':                 32,
            'scaling_factor':        0.7241,
        })()
        cfg = dc_ae_v_f32t4_chunk_causal(model_name, model_path)
        self.ae = DCAEV(cfg).to(dtype=torch.bfloat16)

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.bfloat16

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def encode(self, video):
        return self.ae.encode(video)

    def decode(self, latents, return_dict=True):
        return (self.ae.decode(latents), None)


# ── CLIP vision encoder wrapper ───────────────────────────────────────────────

class _CLIPOutput:
    def __init__(self, features):
        self.hidden_states = (None, features, None)  # hidden_states[-2] = features


class CLIPVisionWrapper(nn.Module):
    """Wraps the bundled XLMRobertaCLIP ViT-H/14 to match diffusers encode_image interface."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        from dc_videogen.wan_blocks.clip import VisionTransformer
        self.vision = VisionTransformer(
            image_size=224, patch_size=14, dim=1280, mlp_ratio=4,
            out_dim=1024, num_heads=16, num_layers=32,
        )
        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        vision_state = {k[len('visual.'):]: v for k, v in state.items() if k.startswith('visual.')}
        missing, unexpected = self.vision.load_state_dict(vision_state, strict=False)
        if missing:
            print(f'[CLIPVisionWrapper] {len(missing)} missing keys')
        if unexpected:
            print(f'[CLIPVisionWrapper] {len(unexpected)} unexpected keys')

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, pixel_values, output_hidden_states=False, **kwargs):
        # Keep in fp32 — custom LayerNorm in VisionTransformer requires fp32 weights.
        pixel_values = pixel_values.to(dtype=next(self.vision.parameters()).dtype)
        features = self.vision(pixel_values)  # [B, 257, 1280]
        return _CLIPOutput(features)


# ── shared text/scheduler loader ─────────────────────────────────────────────

def _load_common(ckpt: pathlib.Path):
    tokenizer    = T5TokenizerFast.from_pretrained(str(ckpt), subfolder='tokenizer')
    text_encoder = UMT5EncoderModel.from_pretrained(
        str(ckpt), subfolder='text_encoder', torch_dtype=torch.bfloat16)
    scheduler    = UniPCMultistepScheduler.from_pretrained(str(ckpt), subfolder='scheduler')
    return tokenizer, text_encoder, scheduler


# ── pipeline builders ─────────────────────────────────────────────────────────

def build_t2v_pipeline() -> DCVideoGenWanTextToVideoPipeline:
    print('[VideoGen] Building T2V pipeline...')
    ckpt = _ensure_videogen_ckpt()
    ae = AEWrapper('dc-ae-v-f32t4c32-1.0-bf16', str(ckpt / 'dc-ae-v-f32t4c32-1.0-bf16.pt'))

    transformer = WanTransformer3DModel.from_pretrained(
        str(ckpt), subfolder='transformer_t2v', torch_dtype=torch.bfloat16,
    )

    tokenizer, text_encoder, scheduler = _load_common(ckpt)

    pipe = DCVideoGenWanTextToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=ae, scheduler=scheduler, transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_i2v_pipeline() -> DCVideoGenWanImageToVideoPipeline:
    print('[VideoGen] Building I2V pipeline...')
    ckpt = _ensure_videogen_ckpt()
    ae = AEWrapper('dc-ae-v-f32t4c32-1.0-bf16', str(ckpt / 'dc-ae-v-f32t4c32-1.0-bf16.pt'))

    transformer = WanTransformer3DModel.from_pretrained(
        str(ckpt), subfolder='transformer_i2v', torch_dtype=torch.bfloat16,
    )

    tokenizer, text_encoder, scheduler = _load_common(ckpt)

    clip_path = str(ckpt / 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth')
    image_encoder = CLIPVisionWrapper(clip_path)  # fp32 — custom LayerNorm requires fp32 weights

    image_processor = CLIPImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        size={'shortest_edge': 224},
        crop_size={'height': 224, 'width': 224},
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        resample=3,
    )

    pipe = DCVideoGenWanImageToVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=ae, scheduler=scheduler,
        image_processor=image_processor, image_encoder=image_encoder,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
