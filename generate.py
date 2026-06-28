#!/usr/bin/env python
"""DC-Gen command-line generation script.

Usage
-----
python generate.py task=<task> prompt="<text>" save_path=<path> [options]

Tasks
-----
  t2i_1k   Text-to-image, up to 1K resolution (DC-AE-f32c32)
  t2i_4k   Text-to-image, up to 4K resolution (DC-AE-1.5-f64c128)
  t2v      Text-to-video 720p (DC-AE-V-f32t4c32)
  i2v      Image-to-video 720p (DC-AE-V-f32t4c32)
  edit     Instruction-based image editing (DC-AE-f32c32)

Options (all have defaults)
---------------------------
  input_image_path=<path>  Required for i2v and edit tasks
  width=<int>              Output width  (default: task-specific)
  height=<int>             Output height (default: task-specific)
  num_frames=<int>         Number of video frames (default: 81)
  steps=<int>              Inference steps (default: 20 for images, 20 for video)
  guidance=<float>         Guidance scale (default: 3.5 for images, 5.0 for video)
  cfg=<float>              True CFG scale for edit task (default: 3.5)
  seed=<int>               RNG seed (default: 42)
  use_expander=yes/no      Expand prompt with Qwen (default: no)

Examples
--------
  python generate.py task=t2i_1k prompt="A cat sitting on a sofa" save_path=out.jpg
  python generate.py task=t2i_4k prompt="A mountain at dawn" save_path=out.jpg width=4096 height=4096
  python generate.py task=t2v prompt="A cat surfing" save_path=out.mp4
  python generate.py task=i2v input_image_path=cat.jpg prompt="The cat waves" save_path=out.mp4
  python generate.py task=edit input_image_path=photo.jpg prompt="Make it snowy" save_path=edit.png
  python generate.py task=t2i_1k prompt="一只猫" save_path=out.jpg use_expander=yes
"""

from __future__ import annotations

import math
import os
import pathlib
import sys
import time
import uuid

import torch
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image

repo_root = pathlib.Path(__file__).resolve().parent

from pipeline_dcgen_flux import DCGen_FluxPipeline
from pipeline_dc_videogen import build_t2v_pipeline, build_i2v_pipeline
from pipeline_dc_qwen_edit import build_qwen_edit_pipeline

HUB_REPO     = 'nvidia/DC-Gen-FLUX.1-Krea-Dev'
HUB_REPO_1K  = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res1K'
HUB_REPO_4K  = 'DC-Gen-FLUX.1-Krea-Dev-v1.0-Res4K'

VIDEO_NEGATIVE_PROMPT = (
    'Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, '
    'static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, '
    'extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, '
    'fused fingers, still picture, messy background, three legs, many people in the background, '
    'walking backwards'
)

# ── Pipeline loaders ──────────────────────────────────────────────────────────

def _download_subdir(subdir: str) -> pathlib.Path:
    local_dir = repo_root / 'pretrained_models' / subdir
    if not (local_dir / 'model_index.json').exists():
        token = os.environ.get('HF_TOKEN')
        snapshot_download(
            repo_id=HUB_REPO,
            repo_type='model',
            local_dir=str(local_dir),
            allow_patterns=f'{subdir}/*',
            token=token,
        )
        nested = local_dir / subdir
        if nested.exists():
            for item in nested.iterdir():
                item.rename(local_dir / item.name)
            nested.rmdir()
    return local_dir


def load_pipe_1k() -> DCGen_FluxPipeline:
    local_dir = _download_subdir(HUB_REPO_1K)
    pipe = DCGen_FluxPipeline.from_pretrained(str(local_dir), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def load_pipe_4k() -> DCGen_FluxPipeline:
    local_dir = _download_subdir(HUB_REPO_4K)
    pipe = DCGen_FluxPipeline.from_pretrained(str(local_dir), torch_dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=False)
    return pipe


# ── Prompt expander ───────────────────────────────────────────────────────────

def _is_english(text: str) -> bool:
    for char in text:
        if '一' <= char <= '鿿':   # CJK
            return False
        if '぀' <= char <= 'ヿ':   # Hiragana / Katakana
            return False
        if '가' <= char <= '힯':   # Hangul
            return False
    return True


def expand_prompt(prompt: str, mode: str, image: Image.Image | None = None) -> str:
    """Translate non-English input then expand with the appropriate Qwen expander."""
    from qwen_25_extend import QwenPromptExpander, TRANSLATE_TEXT_SYS_PROMPT

    dev = 'cuda'

    # Step 1: translate to English if needed
    if not _is_english(prompt):
        print('[Expander] Translating prompt to English...')
        translator = QwenPromptExpander(is_t2i=True)
        translator.to(dev)
        try:
            prompt = translator.extend(prompt, TRANSLATE_TEXT_SYS_PROMPT).prompt
            print(f'[Expander] Translated: {prompt}')
        finally:
            translator.to('cpu')
            torch.cuda.empty_cache()

    # Step 2: expand
    print('[Expander] Expanding prompt...')
    if mode in ('t2i_1k', 't2i_4k'):
        expander = QwenPromptExpander(is_t2i=True)
    elif mode == 't2v':
        expander = QwenPromptExpander(is_t2v=True)
    elif mode == 'i2v':
        expander = QwenPromptExpander(is_i2v=True)
    elif mode == 'edit':
        expander = QwenPromptExpander(is_vl=True, is_edit=True)
    else:
        return prompt

    expander.to(dev)
    try:
        result = expander(prompt, tar_lang='en', image=image)
        expanded = result.prompt
    finally:
        expander.to('cpu')
        torch.cuda.empty_cache()

    print(f'[Expander] Extended prompt:\n{expanded}')
    return expanded


# ── Task implementations ──────────────────────────────────────────────────────

def run_t2i_1k(prompt, save_path, width, height, steps, guidance, seed, use_expander):
    w = width or 1024
    h = height or 1024
    print(f'[t2i_1k] {h}×{w}, {steps} steps, seed={seed}')
    if use_expander:
        prompt = expand_prompt(prompt, 't2i_1k')

    pipe = load_pipe_1k()
    pipe.to('cuda')
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pipe(
            prompt.strip(),
            height=h, width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(seed),
            use_flux_2=True,
        ).images[0]
    print(f'[t2i_1k] Generated in {time.perf_counter() - t0:.1f}s')
    pipe.to('cpu')
    torch.cuda.empty_cache()
    out.save(save_path)
    print(f'[t2i_1k] Saved → {save_path}')


def run_t2i_4k(prompt, save_path, width, height, steps, guidance, seed, use_expander):
    w = width or 4096
    h = height or 4096
    print(f'[t2i_4k] {h}×{w}, {steps} steps, seed={seed}')
    if use_expander:
        prompt = expand_prompt(prompt, 't2i_4k')

    pipe = load_pipe_4k()
    pipe.to('cuda')
    t0 = time.perf_counter()
    with torch.no_grad():
        out = pipe(
            prompt.strip(),
            height=h, width=w,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(seed),
            use_flux_2=False,
        ).images[0]
    print(f'[t2i_4k] Generated in {time.perf_counter() - t0:.1f}s')
    pipe.to('cpu')
    torch.cuda.empty_cache()
    out.save(save_path)
    print(f'[t2i_4k] Saved → {save_path}')


def run_t2v(prompt, save_path, num_frames, steps, guidance, seed, use_expander):
    h, w = 720, 1280
    print(f'[t2v] {h}×{w}, {num_frames} frames, {steps} steps, seed={seed}')
    if use_expander:
        prompt = expand_prompt(prompt, 't2v')

    pipe = build_t2v_pipeline()
    pipe.to('cuda')
    t0 = time.perf_counter()

    _timing = {}
    def _step_cb(pipeline, i, t, cb_kwargs):
        if i == 0:
            torch.cuda.synchronize()
            _timing['cond_end'] = time.perf_counter()
        return cb_kwargs

    with torch.no_grad():
        out = pipe(
            prompt=prompt.strip(),
            negative_prompt=VIDEO_NEGATIVE_PROMPT,
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(seed),
            output_type='latent',
            callback_on_step_end=_step_cb,
            callback_on_step_end_tensor_inputs=['latents'],
        )
        latents = out.frames.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video_tensor = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video_tensor, output_type='np')
    print(f'[t2v] Generated in {time.perf_counter() - t0:.1f}s')
    pipe.to('cpu')
    torch.cuda.empty_cache()
    export_to_video(video[0], save_path, fps=16)
    print(f'[t2v] Saved → {save_path}')


def run_i2v(image_path, prompt, save_path, num_frames, steps, guidance, seed, use_expander):
    img = Image.open(image_path).convert('RGB')
    print(f'[i2v] input={image_path}, {num_frames} frames, {steps} steps, seed={seed}')
    if use_expander:
        prompt = expand_prompt(prompt, 'i2v', image=img)

    pipe = build_i2v_pipeline()
    mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    aspect = img.height / img.width
    h = round(math.sqrt(720 * 1280 * aspect)) // mod * mod
    w = round(math.sqrt(720 * 1280 / aspect)) // mod * mod
    img = img.resize((w, h))

    pipe.to('cuda')
    t0 = time.perf_counter()

    _timing = {}
    def _step_cb(pipeline, i, t, cb_kwargs):
        if i == 0:
            torch.cuda.synchronize()
            _timing['cond_end'] = time.perf_counter()
        return cb_kwargs

    with torch.no_grad():
        out = pipe(
            image=img,
            prompt=prompt.strip(),
            negative_prompt=VIDEO_NEGATIVE_PROMPT,
            height=h, width=w,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=torch.Generator('cuda').manual_seed(seed),
            output_type='latent',
            callback_on_step_end=_step_cb,
            callback_on_step_end_tensor_inputs=['latents'],
        )
        latents = out.frames.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        video_tensor = pipe.vae.decode(latents, return_dict=False)[0]
        video = pipe.video_processor.postprocess_video(video_tensor, output_type='np')
    print(f'[i2v] Generated in {time.perf_counter() - t0:.1f}s')
    pipe.to('cpu')
    torch.cuda.empty_cache()
    export_to_video(video[0], save_path, fps=16)
    print(f'[i2v] Saved → {save_path}')


def run_edit(image_path, prompt, save_path, steps, cfg, seed, use_expander):
    from pipeline_qwen_image_edit import calculate_dimensions
    img = Image.open(image_path).convert('RGB')
    print(f'[edit] input={image_path}, {steps} steps, cfg={cfg}, seed={seed}')
    if use_expander:
        prompt = expand_prompt(prompt, 'edit', image=img)

    pipe = build_qwen_edit_pipeline()
    _cw, _ch, _ = calculate_dimensions(1024 * 1024, img.size[0] / img.size[1])
    _mul = pipe.vae_scale_factor * 2
    _w = _cw // _mul * _mul
    _h = _ch // _mul * _mul

    pipe.to('cuda')
    t0 = time.perf_counter()

    _timing = {}
    def _step_cb(pipeline, i, t, cb_kwargs):
        if i == 0:
            torch.cuda.synchronize()
            _timing['cond_end'] = time.perf_counter()
        return cb_kwargs

    with torch.inference_mode():
        out = pipe(
            image=img,
            prompt=prompt.strip(),
            negative_prompt=' ',
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            height=_h, width=_w,
            generator=torch.Generator('cuda').manual_seed(seed),
            output_type='latent',
            callback_on_step_end=_step_cb,
            callback_on_step_end_tensor_inputs=['latents'],
        )
        latents = pipe._unpack_latents(out.images, _h, _w, pipe.vae_scale_factor)
        latents = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor
        decoded = pipe.vae.decode(latents, return_dict=False)[0]
        out_img = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
    print(f'[edit] Generated in {time.perf_counter() - t0:.1f}s')
    pipe.to('cpu')
    torch.cuda.empty_cache()
    out_img.save(save_path)
    print(f'[edit] Saved → {save_path}')


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_kwargs(argv):
    kw = {}
    for arg in argv:
        if '=' not in arg:
            print(f'[Error] Unrecognised argument: {arg!r}  (expected key=value)')
            sys.exit(1)
        k, _, v = arg.partition('=')
        kw[k.strip()] = v.strip()
    return kw


def main():
    kw = parse_kwargs(sys.argv[1:])

    task = kw.get('task', '').lower()
    if task not in ('t2i_1k', 't2i_4k', 't2v', 'i2v', 'edit'):
        print(__doc__)
        sys.exit(1)

    prompt      = kw.get('prompt', '')
    save_path   = kw.get('save_path', f'output_{task}_{uuid.uuid4().hex[:6]}.{"mp4" if task in ("t2v","i2v") else "jpg" if task in ("t2i_1k","t2i_4k") else "png"}')
    img_path    = kw.get('input_image_path', None)
    width       = int(kw['width'])  if 'width'  in kw else None
    height      = int(kw['height']) if 'height' in kw else None
    num_frames  = int(kw.get('num_frames', 81))
    seed        = int(kw.get('seed', 42))
    use_exp     = kw.get('use_expander', 'no').lower() in ('yes', 'true', '1')

    if task == 't2i_1k':
        steps    = int(kw.get('steps', 20))
        guidance = float(kw.get('guidance', 3.5))
        run_t2i_1k(prompt, save_path, width, height, steps, guidance, seed, use_exp)

    elif task == 't2i_4k':
        steps    = int(kw.get('steps', 20))
        guidance = float(kw.get('guidance', 3.5))
        run_t2i_4k(prompt, save_path, width, height, steps, guidance, seed, use_exp)

    elif task == 't2v':
        steps    = int(kw.get('steps', 20))
        guidance = float(kw.get('guidance', 5.0))
        run_t2v(prompt, save_path, num_frames, steps, guidance, seed, use_exp)

    elif task == 'i2v':
        if not img_path:
            print('[Error] i2v requires input_image_path=<path>')
            sys.exit(1)
        steps    = int(kw.get('steps', 20))
        guidance = float(kw.get('guidance', 5.0))
        run_i2v(img_path, prompt, save_path, num_frames, steps, guidance, seed, use_exp)

    elif task == 'edit':
        if not img_path:
            print('[Error] edit requires input_image_path=<path>')
            sys.exit(1)
        steps = int(kw.get('steps', 30))
        cfg   = float(kw.get('cfg', 3.5))
        run_edit(img_path, prompt, save_path, steps, cfg, seed, use_exp)


if __name__ == '__main__':
    main()
