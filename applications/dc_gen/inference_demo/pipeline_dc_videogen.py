"""
DC-VideoGen pipeline builders — T2V and I2V.
The T2V/I2V pipeline classes and the DC-AE-V autoencoder live in this package.
Checkpoints are auto-downloaded from HuggingFace on first use.
"""

import os

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download

from .dc_ae_v import DCAEV, dc_ae_v_f32t4_chunk_causal
from .pipeline_dc_videogen_wan_i2v import DCVideoGenWanImageToVideoPipeline
from .pipeline_dc_videogen_wan_t2v import DCVideoGenWanTextToVideoPipeline

# ── HuggingFace repos ─────────────────────────────────────────────────────────
HUB_REPO_T2V = "dc-ai/DC-Gen-Wan2.1-T2V-14B"
HUB_REPO_I2V = "dc-ai/DC-Gen-Wan2.1-I2V-14B"
HUB_REPO_I2V_IMAGE_ENCODER = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

# ── local cache (mirrors pretrained_models/ pattern used for image pipelines) ─
PRETRAINED_MODELS_DIR = os.path.join("applications", "dc_gen", "inference_demo", "pretrained_models")
CKPT_T2V = os.path.join(PRETRAINED_MODELS_DIR, "DC-Gen-Wan2.1-T2V-14B")
CKPT_I2V = os.path.join(PRETRAINED_MODELS_DIR, "DC-Gen-Wan2.1-I2V-14B")

_REQUIRED_T2V_PATHS = (
    "dc-ae-v-f32t4c32-1.0-bf16.pt",
    "transformer/config.json",
)
_REQUIRED_I2V_PATHS = _REQUIRED_T2V_PATHS


def _ensure_ckpt(hub_repo: str, ckpt: str, required: tuple[str, ...]) -> str:
    """Download checkpoints from HF if any required file is missing."""
    if not all(os.path.isfile(os.path.join(ckpt, relative_path)) for relative_path in required):
        missing = [relative_path for relative_path in required if not os.path.isfile(os.path.join(ckpt, relative_path))]
        print(f"[VideoGen] Missing checkpoints: {missing}")
        print(f"[VideoGen] Downloading from {hub_repo} ...")
        os.makedirs(ckpt, exist_ok=True)
        token = os.environ.get("HF_TOKEN")
        snapshot_download(
            repo_id=hub_repo,
            repo_type="model",
            local_dir=ckpt,
            token=token,
        )
    return ckpt


from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
from transformers import CLIPImageProcessor, CLIPVisionModel, T5TokenizerFast, UMT5EncoderModel

# ── VAE wrapper ───────────────────────────────────────────────────────────────


class AEWrapper(nn.Module):
    def __init__(self, model_name: str, model_path: str):
        super().__init__()
        self.config = type(
            "C",
            (),
            {
                "scale_factor_temporal": 4,
                "scale_factor_spatial": 32,
                "z_dim": 32,
                "scaling_factor": 0.7241,
            },
        )()
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
            return torch.device("cpu")

    def encode(self, video):
        return self.ae.encode(video)

    def decode(self, latents, return_dict=True):
        return (self.ae.decode(latents), None)


# ── shared text/scheduler loader ─────────────────────────────────────────────


def _load_common(ckpt: str):
    tokenizer = T5TokenizerFast.from_pretrained(ckpt, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    scheduler = UniPCMultistepScheduler.from_pretrained(ckpt, subfolder="scheduler")
    return tokenizer, text_encoder, scheduler


# ── pipeline builders ─────────────────────────────────────────────────────────


def build_t2v_pipeline() -> DCVideoGenWanTextToVideoPipeline:
    print("[VideoGen] Building T2V pipeline...")
    ckpt = _ensure_ckpt(HUB_REPO_T2V, CKPT_T2V, _REQUIRED_T2V_PATHS)
    ae = AEWrapper("dc-ae-v-1.0-f32t4c32-bf16", os.path.join(ckpt, "dc-ae-v-f32t4c32-1.0-bf16.pt"))

    transformer = WanTransformer3DModel.from_pretrained(
        ckpt,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    tokenizer, text_encoder, scheduler = _load_common(ckpt)

    pipe = DCVideoGenWanTextToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=ae,
        scheduler=scheduler,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def build_i2v_pipeline() -> DCVideoGenWanImageToVideoPipeline:
    print("[VideoGen] Building I2V pipeline...")
    ckpt = _ensure_ckpt(HUB_REPO_I2V, CKPT_I2V, _REQUIRED_I2V_PATHS)
    ae = AEWrapper("dc-ae-v-1.0-f32t4c32-bf16", os.path.join(ckpt, "dc-ae-v-f32t4c32-1.0-bf16.pt"))

    transformer = WanTransformer3DModel.from_pretrained(
        ckpt,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    tokenizer, text_encoder, scheduler = _load_common(ckpt)

    image_encoder = CLIPVisionModel.from_pretrained(
        HUB_REPO_I2V_IMAGE_ENCODER,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    # Preserve the existing resize-and-center-crop behavior; the official Wan
    # processor resizes directly to 224x224 and changes the conditioning input.
    image_processor = CLIPImageProcessor(
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        size={"shortest_edge": 224},
        crop_size={"height": 224, "width": 224},
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        resample=3,
    )

    pipe = DCVideoGenWanImageToVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=ae,
        scheduler=scheduler,
        image_processor=image_processor,
        image_encoder=image_encoder,
        transformer=transformer,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
