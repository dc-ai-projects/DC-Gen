# DC-Gen Inference Demo

Command-line generation scripts for text-to-image (1K and 4K), text-to-video, image-to-video, instruction-based image editing, and standalone DC-AE-V reconstruction. Run every command below **from the repository root.**

## Usage

```bash
python -m applications.dc_gen.inference_demo.generate task=<task> prompt="<text>" save_path=<path> [options]
```

## Tasks

| Task | Description | VAE |
|------|-------------|-----|
| `t2i_1k` | Text-to-image, 1K resolution | DC-AE-f32c32 |
| `t2i_4k` | Text-to-image, 4K resolution | DC-AE-1.5-f64c128 |
| `t2v` | Text-to-video, 720p | DC-AE-V-f32t4c32 |
| `i2v` | Image-to-video, 720p | DC-AE-V-f32t4c32 |
| `edit` | Instruction-based image editing | DC-AE-f32c32 |

## Options (all have defaults)

| Option | Description |
|--------|-------------|
| `input_image_path=<path>` | Required for i2v and edit tasks |
| `width=<int>` | Image output width; video dimensions are selected automatically |
| `height=<int>` | Image output height; video dimensions are selected automatically |
| `num_frames=<int>` | Video frames: use a positive multiple of **40** (`40`, `80`, `120`, ...); default: 81, rounded down to 80 by DC-AE-V |
| `steps=<int>` | Inference steps (default: 20 for images, 50 for video) |
| `guidance=<float>` | Guidance scale (default: 3.5 for images, 5.0 for video) |
| `cfg=<float>` | True CFG scale for edit task (default: 3.5) |
| `seed=<int>` | RNG seed (default: 42) |
| `use_expander=yes/no` | Expand prompt with Qwen (default: no) |

For `i2v` and `edit` inference with `use_expander=yes`, install the additional vision-language utility:

```bash
pip install qwen_vl_utils
```

## Examples

```bash
# Text-to-image 1K
python -m applications.dc_gen.inference_demo.generate task=t2i_1k prompt="A cat sitting on a sofa" save_path=out.jpg

# Text-to-image 4K
python -m applications.dc_gen.inference_demo.generate task=t2i_4k prompt="A mountain at dawn" save_path=out.jpg width=4096 height=4096

# Text-to-video
python -m applications.dc_gen.inference_demo.generate task=t2v prompt="A cat surfing" save_path=out.mp4

# Image-to-video
python -m applications.dc_gen.inference_demo.generate task=i2v input_image_path=cat.jpg prompt="The cat waves" save_path=out.mp4

# Image editing
python -m applications.dc_gen.inference_demo.generate task=edit input_image_path=photo.jpg prompt="Make it snowy" save_path=edit.png

# Non-English prompt with auto-expand
python -m applications.dc_gen.inference_demo.generate task=t2i_1k prompt="一只猫" save_path=out.jpg use_expander=yes
```

For Wan2.1 download/cache details, video dimensions, and the verified I2V GPU configuration, see [Published Model Inference](../../../projects/DC-Gen/Wan2.1_VideoGen.md#published-model-inference).

## DC-AE-V Reconstruction Demo

See [DC-AE-V Reconstruction and Evaluation](../../../projects/DC-Gen/Wan2.1_VideoGen.md#dc-ae-v-reconstruction-and-evaluation) for checkpoint downloads, all five supported variants, standalone reconstruction commands, and the ActivityNet benchmark recipe. The standalone demo expects an existing checkpoint and your own input MP4. All five variants use 40-frame temporal chunks; explicitly set `t` to a positive multiple of **40**, such as `t=40`, `t=80`, or `t=120`.

Generation model weights are downloaded automatically on first use from Hugging Face Hub into `applications/dc_gen/inference_demo/pretrained_models/`. Wan2.1 I2V also downloads its image encoder through the Hugging Face cache.
