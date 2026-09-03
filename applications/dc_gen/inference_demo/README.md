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
| `width=<int>` | Output width (default: task-specific) |
| `height=<int>` | Output height (default: task-specific) |
| `num_frames=<int>` | Number of video frames (default: 81) |
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

## DC-AE-V Reconstruction Demo

```bash
python -m applications.dc_gen.inference_demo.demo_dc_ae_v \
    model_name=dc-ae-v-1.0-f32t4c32-bf16 \
    model_path=applications/dc_gen/inference_demo/pretrained_models/DC-Gen-Wan2.1-I2V-14B/dc-ae-v-f32t4c32-1.0-bf16.pt \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] \
    h=832 w=480 t=80 \
    run_dir=exp/dc-ae-v-1.0-f32t4c32-bf16
```

Generation model weights are downloaded automatically on first use from HuggingFace Hub into `applications/dc_gen/inference_demo/pretrained_models/`. The standalone DC-AE-V demo expects `model_path` to point to an existing checkpoint.
