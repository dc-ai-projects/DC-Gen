"""
DC-Gen-Qwen-Image-Edit pipeline builder.
Checkpoints auto-downloaded from HuggingFace on first use into
applications/dc_gen/inference_demo/pretrained_models/DC-Gen-Qwen-Image-Edit-Res1K/.
"""

import os
import shutil

import torch
from huggingface_hub import snapshot_download

HUB_REPO_QWEN_EDIT = "dc-ai/DC-Gen-Qwen-Image-Edit-Res1K"

CKPT = os.path.join(
    "applications",
    "dc_gen",
    "inference_demo",
    "pretrained_models",
    "DC-Gen-Qwen-Image-Edit-Res1K",
)

# All of these must exist for the checkpoint to be considered complete.
_REQUIRED = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "transformer/config.json",
    "vae/config.json",
]


def _ckpt_complete() -> bool:
    return all(os.path.isfile(os.path.join(CKPT, relative_path)) for relative_path in _REQUIRED)


def _ensure_qwen_edit_ckpt() -> str:
    if _ckpt_complete():
        return CKPT

    # Previous failed/partial downloads leave garbage that confuses
    # snapshot_download into thinking files are already present.
    # Wiping CKPT forces a clean re-copy from the global HF cache (fast).
    if os.path.exists(CKPT):
        print(f"[QwenEdit] Removing incomplete download at {CKPT} ...")
        shutil.rmtree(CKPT)

    token = os.environ.get("HF_TOKEN")
    print(f"[QwenEdit] Downloading from {HUB_REPO_QWEN_EDIT} ...")
    os.makedirs(CKPT, exist_ok=True)

    snapshot_download(
        repo_id=HUB_REPO_QWEN_EDIT,
        repo_type="model",
        local_dir=CKPT,
        local_dir_use_symlinks=False,
        token=token,
    )

    if not _ckpt_complete():
        raise RuntimeError(
            f"model_index.json (or other required files) not found at {CKPT}. "
            f"Top-level contents: {os.listdir(CKPT)}"
        )
    return CKPT


def build_qwen_edit_pipeline():
    print("[QwenEdit] Building pipeline...")
    ckpt = _ensure_qwen_edit_ckpt()

    from .pipeline_qwen_image_edit import DCQwenImageEditPipeline

    pipe = DCQwenImageEditPipeline.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe
