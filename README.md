# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

DC-Gen adapts high-resolution visual generation and editing models (e.g., FLUX, Wan2.1, Qwen-Image-Edit) to deeply compressed latent spaces through efficient post-training. It enables native 4K image synthesis and achieves up to 54× acceleration.

<p align="center">
  <img src="https://github.com/dc-ai-projects/DC-Gen/raw/main/assets/dc-gen-figures/teaser_page1.png" alt="DC-Gen Teaser" width="100%">
</p>

<p align="center">
  <video src="https://github.com/dc-ai-projects/DC-Gen/raw/main/assets/dc-gen-figures/demo.mp4" autoplay loop muted playsinline controls width="100%"></video>
</p>

---

Command-line interface for running DC-Gen locally. Supports text-to-image (1K and 4K), text-to-video, image-to-video, and instruction-based image editing.

## Setup

```bash
git clone <this-repo>
cd dc-gen-cli
conda create -n dcgen python=3.10 && conda activate dcgen
pip install -r requirements.txt
```

A HuggingFace token with access to `nvidia/` is required. Set it via:

```bash
export HF_TOKEN=<your_hf_token>
```

Model weights are downloaded automatically on first use into `pretrained_models/`.

## Usage

```
python generate.py task=<task> prompt="<text>" save_path=<path> [options]
```

### Tasks

| Task | Description | VAE |
|------|-------------|-----|
| `t2i_1k` | Text-to-image, 1K resolution | DC-AE-f32c32 |
| `t2i_4k` | Text-to-image, 4K resolution | DC-AE-1.5-f64c128 |
| `t2v` | Text-to-video, 720p | DC-AE-V-f32t4c32 |
| `i2v` | Image-to-video, 720p | DC-AE-V-f32t4c32 |
| `edit` | Instruction-based image editing | DC-AE-f32c32 |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `input_image_path` | — | Input image path (required for `i2v` and `edit`) |
| `width` | task default | Output width in pixels |
| `height` | task default | Output height in pixels |
| `num_frames` | `81` | Number of video frames (`t2v`, `i2v`) |
| `steps` | `20` (images) / `20` (video) / `30` (edit) | Inference steps |
| `guidance` | `3.5` (images) / `5.0` (video) | Guidance scale |
| `cfg` | `3.5` | True CFG scale (edit only) |
| `seed` | `42` | Random seed |
| `use_expander` | `no` | Expand prompt with Qwen (`yes`/`no`). Supports non-English input. |

### Examples

```bash
# Text-to-image 1K
python generate.py task=t2i_1k prompt="A cat lazily lying in a dog's arms in the sun" save_path=out.jpg

# Text-to-image 4K
python generate.py task=t2i_4k prompt="A mountain lake at dawn, mist rising" save_path=out.jpg width=4096 height=4096

# Text-to-video
python generate.py task=t2v prompt="A white cat wearing sunglasses surfs on a wave" save_path=out.mp4

# Image-to-video
python generate.py task=i2v input_image_path=cat.jpg prompt="The cat waves its paw" save_path=out.mp4

# Image editing
python generate.py task=edit input_image_path=photo.jpg prompt="Make the background snowy" save_path=edit.png

# Non-English prompt with auto-expand
python generate.py task=t2i_1k prompt="一只猫慵懒地躺在一只狗的怀里晒太阳。" save_path=out.jpg use_expander=yes
```
