# DC-Gen: Video Generation with Wan2.1

This guide covers published DC-Gen-Wan2.1 inference, standalone DC-AE-V reconstruction, and 480p adaptation and evaluation recipes. It consolidates the former `projects/DC-VideoGen` guides. Run all commands from the repository root unless a command explicitly changes directory.

- [Environment](#environment)
- [Published Model Inference](#published-model-inference)
- [DC-AE-V Reconstruction and Evaluation](#dc-ae-v-reconstruction-and-evaluation)
- [Prepare Training Checkpoints](#prepare-training-checkpoints)
- [Prepare Evaluation Benchmark](#prepare-evaluation-benchmark)
- [Test Original Wan2.1 Models](#test-original-wan21-models)
- [Prepare Training Data](#prepare-training-data)
- [Train T2V-1.3B](#model-1-dc-gen-wan21-t2v-13b), [T2V-14B](#model-2-dc-gen-wan21-t2v-14b), or [I2V-14B](#model-3-dc-gen-wan21-i2v-14b)
- [Evaluate Adapted Checkpoints](#evaluate-dc-gen-model)

## Environment

Follow [Getting Started](../../README.md#getting-started) to install PyTorch, the repository requirements, and FlashAttention in a CUDA environment. The inference examples require an NVIDIA GPU with BF16 support. Replace `path/to/input.jpg` and `path/to/input.mp4` below with your own files; these inputs are not bundled with the repository.

The training and trainer-based reconstruction/evaluation commands also use the following dependencies from the former DC-VideoGen setup:

```bash
pip install numpy==1.26.4 easydict fairscale
```

For VBench scoring, install its additional dependencies:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
pip install git+https://github.com/WenkunHe/VBench.git --no-build-isolation
```

## Published Model Inference

These commands automatically download the published [DC-Gen-Wan2.1-T2V-14B](https://huggingface.co/dc-ai/DC-Gen-Wan2.1-T2V-14B) and [DC-Gen-Wan2.1-I2V-14B](https://huggingface.co/dc-ai/DC-Gen-Wan2.1-I2V-14B) models. They do not require the checkpoint conversion, training data, or VBench setup later in this guide. See the [inference CLI reference](../../applications/dc_gen/inference_demo/README.md) for all tasks and options.

```bash
# Image-to-video
python -m applications.dc_gen.inference_demo.generate \
    task=i2v input_image_path=path/to/input.jpg \
    prompt="The subject moves gently while the camera slowly pans right." \
    save_path=i2v.mp4 steps=50 num_frames=80 guidance=5.0 seed=42 use_expander=no

# Text-to-video
python -m applications.dc_gen.inference_demo.generate \
    task=t2v prompt="A cat surfing on a sunny day." \
    save_path=t2v.mp4 steps=50 num_frames=80 guidance=5.0 seed=42 use_expander=no
```

For both T2V and I2V, set `num_frames` to a positive multiple of **40**, such as `40`, `80`, or `120`, to match DC-AE-V temporal chunks. For a quick I2V smoke test, use the same I2V command with `steps=2 num_frames=40`. This only checks execution, not generation quality.

Videos are saved at 16 FPS. I2V derives its dimensions from the input aspect ratio using a 720 × 1280 pixel budget and rounds each dimension to a multiple of 32; a square image produces 960 × 960 output. The `width` and `height` CLI options do not override video dimensions.

Model files are cached under `applications/dc_gen/inference_demo/pretrained_models/`. I2V also loads the `image_encoder` subfolder of `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers` through the Hugging Face cache.

## DC-AE-V Reconstruction and Evaluation

### Download AE Checkpoints

The Wan2.1 adaptation recipes use `dc-ae-v-1.0-f32t4c32-bf16`. Download it directly into the directory expected by the model registry:

```bash
hf download dc-ai/dc-ae-v-1.0-f32t4c32 dc-ae-v-1.0-f32t4c32-bf16.pt \
    --local-dir assets/checkpoints/dc_gen/dc_ae_v
```

The other four supported 1.0 variants are also available on Hugging Face. Download only the variant you need; these files already use the names expected by the model registry:

```bash
hf download dc-ai/dc-ae-v-1.0-f32t4c64 dc-ae-v-1.0-f32t4c64-bf16.pt \
    --local-dir assets/checkpoints/dc_gen/dc_ae_v

hf download dc-ai/dc-ae-v-1.0-f32t4c128 dc-ae-v-1.0-f32t4c128-bf16.pt \
    --local-dir assets/checkpoints/dc_gen/dc_ae_v

hf download dc-ai/dc-ae-v-1.0-f32t4c256 dc-ae-v-1.0-f32t4c256-bf16.pt \
    --local-dir assets/checkpoints/dc_gen/dc_ae_v

hf download dc-ai/dc-ae-v-1.0-f64t4c128 dc-ae-v-1.0-f64t4c128-bf16.pt \
    --local-dir assets/checkpoints/dc_gen/dc_ae_v
```

Use the matching model name and scaling factor below:

| Model name | Spatial / temporal compression | Latent channels | Scaling factor for the trainer autoencoder wrapper |
| --- | --- | --- | --- |
| [`dc-ae-v-1.0-f32t4c32-bf16`](https://huggingface.co/dc-ai/dc-ae-v-1.0-f32t4c32) | 32 / 4 | 32 | 0.7241 |
| [`dc-ae-v-1.0-f32t4c64-bf16`](https://huggingface.co/dc-ai/dc-ae-v-1.0-f32t4c64) | 32 / 4 | 64 | 0.7614 |
| [`dc-ae-v-1.0-f32t4c128-bf16`](https://huggingface.co/dc-ai/dc-ae-v-1.0-f32t4c128) | 32 / 4 | 128 | 0.4457 |
| [`dc-ae-v-1.0-f32t4c256-bf16`](https://huggingface.co/dc-ai/dc-ae-v-1.0-f32t4c256) | 32 / 4 | 256 | 0.4752 |
| [`dc-ae-v-1.0-f64t4c128-bf16`](https://huggingface.co/dc-ai/dc-ae-v-1.0-f64t4c128) | 64 / 4 | 128 | 1.5586 |

All five DC-AE-V 1.0 variants use 40-frame temporal chunks, each corresponding to 10 latent frames. The supported frame counts are positive multiples of **40** (`40`, `80`, `120`, ...). The temporal compression ratio of 4 in the table describes the input-to-latent frame ratio; frame counts must also align with the 40-frame chunks.

These are latent normalization settings calibrated for the model and data distribution. Keep `0.7241` for the published Wan models to match their training normalization. The `f64t4c128` factor is `1.5586`; the `0.4752` value belongs to `f32t4c256`.

The published video inference bundles use a different filename, `dc-ae-v-f32t4c32-1.0-bf16.pt`. The standalone demo accepts that file through `model_path`; registry-based commands use the canonical filename above. Model names always use the `dc-ae-v-1.0-...` ordering.

### Reconstruct a Video

Use the standalone demo to encode and decode an MP4 with any model from the table. Set `model_name` and `model_path` to the matching model and checkpoint:

```bash
python -m applications.dc_gen.inference_demo.demo_dc_ae_v \
    model_name=dc-ae-v-1.0-f32t4c32-bf16 \
    model_path=assets/checkpoints/dc_gen/dc_ae_v/dc-ae-v-1.0-f32t4c32-bf16.pt \
    input_path_list=[path/to/input.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_dc_ae_v/dc-ae-v-1.0-f32t4c32-bf16
```

The output is a reconstructed MP4 in `run_dir`, saved at 8 FPS. Choose spatial dimensions divisible by the model's spatial compression ratio and explicitly set `t` to a positive multiple of **40**, such as `t=40`, `t=80`, or `t=120`. The standalone demo operates on raw AE latents and does not need a scaling-factor argument.

To save input and reconstruction together using the trainer autoencoder wrapper, use the original reconstruction entrypoint. For another model, change both `autoencoder.name` and its scaling factor according to the table:

```bash
python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 \
    input_path_list=[path/to/input.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f32t4c32-bf16
```

### Evaluate AE Reconstruction

Run all commands below from the repository root. Install the Python requirements from [Environment](#environment) and ensure the `ffmpeg` executable is on `PATH`. Dataset preparation runs on CPUs; FVD reference generation and AE evaluation require GPUs. The conversion commands use one Ray worker plus one coordinator, so provide at least two CPU slots. Increase `num_processes` only when at least `num_processes + 1` CPU slots are available.

#### Prepare ActivityNet 1.2 Test

The [ActivityNet mirror](https://huggingface.co/datasets/YimuWang/ActivityNet) stores the test archives in parts. Download and concatenate the 1.2 parts under `assets/dataset/activitynet`, verify the historical MD5 checksum, and then extract and convert the videos. The subshell stops on an error without changing the parent shell's working directory.

```bash
(
    set -e
    mkdir -p assets/dataset/activitynet
    hf download --repo-type dataset YimuWang/ActivityNet \
        --include "v1-2_test.tar.gz*" --local-dir assets/dataset/activitynet
    cat assets/dataset/activitynet/v1-2_test.tar.gz.* > assets/dataset/activitynet/v1-2_test.tar.gz
    echo "b06ca628443da1c1389c69b2feac6ca2  assets/dataset/activitynet/v1-2_test.tar.gz" | md5sum -c -
    tar -xzf assets/dataset/activitynet/v1-2_test.tar.gz -C assets/dataset/activitynet

    RAY_DEDUP_LOGS=0 python -m scripts.data.convert_activitynet_to_mp4 \
        data_dir=assets/dataset/activitynet/v1-2/test num_processes=1
    python -m dc_ai.staecore.data_provider.examine \
        dataset=ActivityNet12Test use_ray=False run_dir=exp/examine/ActivityNet12Test
    python -m dc_ai.staecore.data_provider.examine \
        dataset=ActivityNet12Test merge_results=True run_dir=exp/examine/ActivityNet12Test
)
```

This writes `assets/data/examination/ActivityNet12Test.csv`. The converter preserves existing MP4 files and converts MKV/WebM files with FFmpeg. The data loader only selects MP4 files, so the original files can remain in place.

#### Prepare ActivityNet 1.3 Test

The 1.3 test set depends on the 1.2 test videos prepared above. Convert the 1.3 archive, add relative symlinks to the 1.2 MP4 files, and generate metadata after the directory is complete. The linking step preserves existing files and can be repeated. Keep both `v1-2` and `v1-3` directories together so the symlinks remain valid.

```bash
(
    set -e
    hf download --repo-type dataset YimuWang/ActivityNet \
        --include "v1-3_test.tar.gz*" --local-dir assets/dataset/activitynet
    cat assets/dataset/activitynet/v1-3_test.tar.gz.* > assets/dataset/activitynet/v1-3_test.tar.gz
    echo "ec218c66991d776ad0ff826c0b43836d  assets/dataset/activitynet/v1-3_test.tar.gz" | md5sum -c -
    tar -xzf assets/dataset/activitynet/v1-3_test.tar.gz -C assets/dataset/activitynet

    RAY_DEDUP_LOGS=0 python -m scripts.data.convert_activitynet_to_mp4 \
        data_dir=assets/dataset/activitynet/v1-3/test num_processes=1
    (
        cd assets/dataset/activitynet/v1-3/test
        for source_video in ../../v1-2/test/*.mp4; do
            if [ ! -f "$source_video" ]; then
                echo "Missing ActivityNet 1.2 MP4 files; complete the 1.2 preparation first." >&2
                exit 1
            fi
            if [ ! -e "${source_video##*/}" ]; then
                ln -s "$source_video" .
            fi
        done
    )
    python -m dc_ai.staecore.data_provider.examine \
        dataset=ActivityNet13Test use_ray=False run_dir=exp/examine/ActivityNet13Test
    python -m dc_ai.staecore.data_provider.examine \
        dataset=ActivityNet13Test merge_results=True run_dir=exp/examine/ActivityNet13Test
)
```

This writes `assets/data/examination/ActivityNet13Test.csv`. Metadata rows follow the sorted MP4 paths; if you change the dataset contents or symlinks, regenerate metadata using a fresh `run_dir` for both scanning and merging before generating reference statistics or evaluating models.

#### Generate FVD Reference Statistics

Generate the reference statistics once from the prepared ActivityNet 1.3 test set. Create the I3D cache directory before the first run; the metric downloads `i3d_torchscript.pt` there if it is missing. `mean=0.0 std=1.0` supplies videos in the `[0, 1]` range expected by the FVD metric. `fps=null` keeps the source frame rate and corresponds to `None` in the output filename.

```bash
mkdir -p assets/checkpoints/pretrained assets/data/fvd
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.staecore.generate_reference \
    dataset=activitynet_1_3_test \
    activitynet_1_3_test.h=256 activitynet_1_3_test.w=256 activitynet_1_3_test.t=80 \
    activitynet_1_3_test.fps=null activitynet_1_3_test.mean=0.0 activitynet_1_3_test.std=1.0 \
    activitynet_1_3_test.batch_size=1 \
    fvd.save_path=assets/data/fvd/activitynet_1_3_test_256_256_80_None.npz
```

#### Run AE Evaluation

After generating the 80-frame reference, evaluate latent statistics, PSNR, SSIM, LPIPS, and FVD:

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.staecore.trainer \
    mode=eval amp=bf16 \
    eval_data_providers=[ActivityNet13Test_256_256_80] base_sample_size=[256,256,16] base_batch_size=4 \
    model=dc-ae-v-1.0-f32t4c32-bf16 compute_fvd=True \
    run_dir=exp/evaluate_st_ae/dc-ae-v-1.0-f32t4c32-bf16
```

The provider reads `assets/data/fvd/activitynet_1_3_test_256_256_80_None.npz` automatically. Set `compute_fvd=False` to skip FVD and its reference-generation step; the prepared dataset and metadata are still required.

Expected results:

| Latent mean | Latent RMS | Inverse RMS | Latent std | PSNR | SSIM | LPIPS | FVD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0086 | 1.3286 | 0.7527 | 1.3286 | 31.0825 | 0.9009 | 0.0447 | 13.7873 |

## Prepare Training Checkpoints

The remaining sections use the original Wan checkpoints and the repository's trainer to adapt and evaluate models at 480p. The parameters are debugging recipes; refer to Appendix E of the [DC-Gen paper](https://arxiv.org/abs/2509.25180) for the training hyperparameters used in the paper. Complete the [AE checkpoint download](#download-ae-checkpoints) first, then prepare the encoders and null embeddings before converting the diffusion checkpoints. Only the checkpoint for the model you intend to train is needed.

### Prepare Text and Image Encoders

Both T2V and I2V use the same T5 weights. The text encoder reads them from `assets/checkpoints/t2v`, not a model-size subdirectory. Its tokenizer is downloaded from `google/umt5-xxl` on first use.

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir assets/checkpoints/t2v

# Additional image encoder for I2V
hf download Wan-AI/Wan2.1-I2V-14B-480P \
    models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --local-dir assets/checkpoints/i2v
```

### Prepare Null Embeddings

Run these on a GPU before checkpoint conversion, which reads the resulting files:

```bash
python -m applications.dc_videogen.get_null_embedding id=wan2.1-t2v/umt5-512-bf16
python -m applications.dc_videogen.get_null_embedding id=wan2.1-i2v/umt5-512-bf16
```

The expected outputs are `assets/data/null_text_embeddings/wan2.1-t2v/umt5-512-bf16.pth` and `assets/data/null_text_embeddings/wan2.1-i2v/umt5-512-bf16.pth`.

### Convert Wan2.1-T2V-1.3B

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B \
    diffusion_pytorch_model.safetensors \
    --local-dir assets/checkpoints/t2v/1.3b
python -m applications.dc_videogen.convert_checkpoint model=wan_t2v_1.3B \
    save_filename=wan_t2v_1.3b_pretrained.pt
rm assets/checkpoints/t2v/1.3b/diffusion_pytorch_model.safetensors
```

### Convert Wan2.1-T2V-14B

```bash
hf download Wan-AI/Wan2.1-T2V-14B --include "diffusion_pytorch_model-*.safetensors" --local-dir assets/checkpoints/t2v/14b
python -m applications.dc_videogen.convert_checkpoint model=wan_t2v_14B \
    file_dir=assets/checkpoints/t2v/14b \
    save_filename=wan_t2v_14b_pretrained.pt \
    load_filenames=[diffusion_pytorch_model-00001-of-00006.safetensors,diffusion_pytorch_model-00002-of-00006.safetensors,diffusion_pytorch_model-00003-of-00006.safetensors,diffusion_pytorch_model-00004-of-00006.safetensors,diffusion_pytorch_model-00005-of-00006.safetensors,diffusion_pytorch_model-00006-of-00006.safetensors]
rm assets/checkpoints/t2v/14b/diffusion_pytorch_model-*.safetensors
```

### Convert Wan2.1-I2V-14B

```bash
hf download Wan-AI/Wan2.1-I2V-14B-480P --include "diffusion_pytorch_model-*.safetensors" --local-dir assets/checkpoints/i2v/14b
python -m applications.dc_videogen.convert_checkpoint model=wan_i2v_14B \
    file_dir=assets/checkpoints/i2v/14b \
    save_filename=wan_i2v_14b_480p_pretrained.pt \
    load_filenames=[diffusion_pytorch_model-00001-of-00007.safetensors,diffusion_pytorch_model-00002-of-00007.safetensors,diffusion_pytorch_model-00003-of-00007.safetensors,diffusion_pytorch_model-00004-of-00007.safetensors,diffusion_pytorch_model-00005-of-00007.safetensors,diffusion_pytorch_model-00006-of-00007.safetensors,diffusion_pytorch_model-00007-of-00007.safetensors]
rm assets/checkpoints/i2v/14b/diffusion_pytorch_model-*.safetensors
```

## Prepare Evaluation Benchmark

Prepare the corresponding benchmark before training with a VBench evaluation provider or running the final benchmark commands.

### T2V VBench

```bash
mkdir -p assets/data/vbench
wget https://raw.githubusercontent.com/Vchitect/VBench/master/vbench/VBench_full_info.json \
    -O assets/data/vbench/VBench_full_info.json
wget https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt \
    -O assets/data/vbench/vbench_extended.txt
python -m applications.dc_videogen.rewrite_vbench_meta
```

The resulting metadata is `assets/data/vbench/VBench_extended_full_info.json`.

### I2V VBench

Download our processed I2V evaluation inputs from [DC-Gen-Data](https://huggingface.co/datasets/dc-ai/DC-Gen-Data): 355 VBench images resized to 832x480 and the paired extended prompts. These are processed test data derived from [VBench](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v).

```bash
python - <<'PY_DOWNLOAD'
from huggingface_hub import HfFileSystem

HfFileSystem().get(
    "datasets/dc-ai/DC-Gen-Data/eval/wan-i2v/",
    "assets/data/vbench",
    recursive=True,
)
PY_DOWNLOAD
```

The download writes `VBench_i2v_extended_full_info.json` and `vbench_i2v_imgs/832-480/` (355 images) directly under `assets/data/vbench`, matching the existing trainer defaults. The trailing `/` on the remote subfolder keeps its contents at the local destination without adding `eval/wan-i2v/`. Manual-prompt inference does not require this benchmark data.

Two relevant evaluation parameters:

- `skip_vbench_evaluator=False` (default) generates videos and runs VBench scoring. Set it to `True` to generate videos only; benchmark prompts and I2V images are still required.
- `vbench_text_prompt.num_videos_per_prompt=5` and `vbench_image_prompt.num_videos_per_prompt=5` (defaults) generate five videos per prompt when `num_samples` is unset.

## Test Original Wan2.1 Models

These commands use the converted original Wan checkpoints and the original Wan VAE, which is downloaded automatically from Hugging Face on first use.

### Wan2.1-T2V-1.3B

```bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=bf16 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=wan-vae autoencoder_dtype=bf16 \
    wan_t2v.in_channels=16 wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.pretrained_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt wan_t2v.pretrained_source=wan \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 \
    run_dir=exp/demo_videogen/wan2.1_t2v_1.3b
```

### Wan2.1-T2V-14B

```bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=bf16 offload=True \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=wan-vae autoencoder_dtype=bf16 \
    wan_t2v.in_channels=16 wan_t2v.depth=40 wan_t2v.hidden_size=5120 wan_t2v.num_heads=40 wan_t2v.ffn_dim=13824 \
    wan_t2v.pretrained_path=assets/checkpoints/t2v/14b/wan_t2v_14b_pretrained.pt wan_t2v.pretrained_source=wan \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 \
    run_dir=exp/demo_videogen/wan2.1_t2v_14b
```

### Wan2.1-I2V-14B

```bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_i2v resolution=480 amp=bf16 model_dtype=bf16 offload=True \
    image_paths=[assets/fig/wan_i2v_input.JPG] \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] wan_i2v.text_encoder_id=wan2.1-i2v/umt5-512-bf16 \
    autoencoder.name=wan-vae autoencoder_dtype=bf16 \
    wan_i2v.in_channels=16 wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.pretrained_path=assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt wan_i2v.pretrained_source=wan \
    run_dir=exp/demo_videogen/wan2.1_i2v_14b
```

## Prepare Training Data

The recipes below train at 480p using paired Wan and DC-AE-V latents. The supplied step counts are debugging settings, and these training commands have not been revalidated end to end as part of the inference check above.

Download only the Wan training examples from [DC-Gen-Data](https://huggingface.co/datasets/dc-ai/DC-Gen-Data). The supplied `fusionX_480p` TAR shard is example data for the training workflow, not the complete training dataset; its `align/shift_2/` structure is preserved.

```bash
python - <<'PY_DOWNLOAD'
from huggingface_hub import HfFileSystem

HfFileSystem().get(
    "datasets/dc-ai/DC-Gen-Data/train_example/wan/fusionX_480p/",
    "assets/dataset/fusionX_480p",
    recursive=True,
)
PY_DOWNLOAD
```

### Step 1: Generate Meta for RGB Data

```bash
# We only need a small subset for video embedding space alignment
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/fusionX_480p/align
# This is the complete metadata for the dataset
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/fusionX_480p
```

### Step 2: Extract Latent

```bash
# We currently only provide a single tar file. To deal with more tar files, please adjust num_archives_per_task and task_id to support parallel latent feature extraction.

# Wan 2.1
python -m dc_ai.videogencore.data_provider.generate_latent_archives \
    dataset_name=Wan resolution_str=480F32MS num_frames=77 \
    meta_path=assets/dataset/fusionX_480p/align/wids-meta.json original_archive_dir=assets/dataset/fusionX_480p latent_ext=.pth \
    autoencoder.name=wan-vae \
    batch_size=1 num_archives_per_task=1 \
    save_dir=assets/dataset/latent/fusionX_480/wan_21_vae_480

# dc-ae-v-f32t4c32-1.0-bf16
python -m dc_ai.videogencore.data_provider.generate_latent_archives \
    dataset_name=Wan resolution_str=480F32MS num_frames=80 \
    meta_path=assets/dataset/fusionX_480p/wids-meta.json original_archive_dir=assets/dataset/fusionX_480p latent_ext=.pth \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 dtype=bf16 \
    batch_size=1 num_archives_per_task=1 \
    save_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480
```

### Step 3: Generate Meta for Latent

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/wan_21_vae_480/align save_path=assets/data/meta/wids/wan_21_vae_480/fusionX_align.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480 save_path=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480/fusionX.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480/align save_path=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480/fusionX_align.json
```

You can verify the training data by checking paired video reconstructions:

```bash
python -m dc_ai.dc_adapt.data_provider.latent_mixture \
    data_providers=[LatentFusionXAlign] data_ext=.pth \
    teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    teacher_autoencoder.name=wan-vae \
    student_autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 student_autoencoder.scaling_factor=0.7241 \
    batch_size=2 resolution=480F32MS \
    save_checkpoint_steps=10 distributed_method=FSDP
```

## Model 1: DC-Gen-Wan2.1-T2V-1.3B

### Step 1.1: Video Embedding Space Alignment

```bash
python -m dc_ai.dc_adapt.trainer mode=train \
    dataset=latent_video data_type=latent model_type=wan_t2v \
    latent_video.data_providers=[LatentFusionXAlign] latent_video.data_ext=.pth \
    latent_video.teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    latent_video.student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    latent_video.batch_size=4 \
    image_encoder.resolution=480 \
    teacher_stae.name=wan-vae \
    student_stae.name=dc-ae-v-1.0-f32t4c32-bf16 student_stae.scaling_factor=0.7241 \
    teacher_wan_t2v.input_size=[20,60,104] teacher_wan_t2v.patch_size=[1,2,2] \
    teacher_wan_t2v.in_channels=16 teacher_wan_t2v.hidden_size=1536 \
    teacher_wan_t2v.pretrained_source=wan teacher_wan_t2v.pretrained_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt \
    student_wan_t2v.input_size=[20,15,26] student_wan_t2v.patch_size=[1,1,1] \
    student_wan_t2v.in_channels=32 student_wan_t2v.hidden_size=1536 \
    align_type=downsample upsample_mode=nearest \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 \
    optimizer=AdamW lr=2e-4 weight_decay=0 lr_scheduler=WSDCosineLR ema_decay=null betas=[0.9,0.9] amp=bf16 distributed_method=DDP \
    save_checkpoint_steps=1000 save_samples_steps=null eval_steps=5000 max_steps=20000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_1 \
    log=False
```

### Step 1.2: Assemble

```bash
path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_1
checkpoint=step_20000.pt
python -m applications.dc_videogen.wan_t2v_assemble \
    patch_size=[1,1,1] input_size=[20,15,26] in_channels=32 \
    depth=30 hidden_size=1536 num_heads=12 \
    backbone_checkpoint_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/phase_2_init.pt
```

### Step 2: Patch-Head Fine-Tuning

```bash
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_samples=32 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False vbench_text_prompt.batch_size=1 \
    category_ids=[0] save_samples_at_all_ranks=True \
    mixture.resolution=480F32MS mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=bf16 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_1/phase_2_init.pt wan_t2v.freeze_backbone=True \
    optimizer=AdamW lr=0.0001 betas=[0.9,0.999] lr_scheduler=ConstantLRwithWarmup clip_grad=0.1 weight_decay=0.0 warmup_steps=0 warmup_lr=0.0 \
    amp=bf16 activation_checkpointing_mode=transformer \
    max_steps=6000 eval_steps=1000 save_checkpoint_steps=200 save_eval_checkpoint_steps=2000 \
    distributed_method=FSDPWrap \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_2 log=False
```

### Step 3: End-to-End Fine-Tuning

```bash
checkpoint=step_6000.pt
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_samples=32 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False vbench_text_prompt.batch_size=1 \
    category_ids=[0,1,2,3,4,5,6,7,8,9,10] save_samples_at_all_ranks=True \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=bf16 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_2/$checkpoint \
    wan_t2v.freeze_text_embed=True wan_t2v.freeze_cross_attn=True \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 cfg_scale=3.0 \
    use_lora=True lora_rank=256 lora_alpha=512 \
    optimizer=AdamW lr=0.00005 betas=[0.9,0.999] lr_scheduler=ConstantLRwithWarmup warmup_lr=0.0 clip_grad=0.1 weight_decay=0.0 warmup_steps=1000 ema_decay=0.9999 \
    distributed_method=FSDPWrap \
    amp=bf16 activation_checkpointing_mode=transformer \
    max_steps=40000 eval_steps=1000 save_checkpoint_steps=200 save_eval_checkpoint_steps=2000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3 log=False

# merge LoRA weights
python -m applications.dc_videogen.convert_lora_into_base \
    load_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_40000.pt \
    save_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_40000_merged.pt
```

## Model 2: DC-Gen-Wan2.1-T2V-14B

### Step 1.1: Video Embedding Space Alignment

```bash
python -m dc_ai.dc_adapt.trainer mode=train \
    dataset=latent_video data_type=latent model_type=wan_t2v \
    latent_video.data_providers=[LatentFusionXAlign] latent_video.data_ext=.pth \
    latent_video.teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    latent_video.student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    latent_video.batch_size=4 \
    image_encoder.resolution=480 \
    teacher_stae.name=wan-vae \
    student_stae.name=dc-ae-v-1.0-f32t4c32-bf16 student_stae.scaling_factor=0.7241 \
    teacher_wan_t2v.input_size=[20,60,104] teacher_wan_t2v.patch_size=[1,2,2] \
    teacher_wan_t2v.in_channels=16 teacher_wan_t2v.hidden_size=5120 \
    teacher_wan_t2v.pretrained_source=wan teacher_wan_t2v.pretrained_path=assets/checkpoints/t2v/14b/wan_t2v_14b_pretrained.pt \
    student_wan_t2v.input_size=[20,15,26] student_wan_t2v.patch_size=[1,1,1] \
    student_wan_t2v.in_channels=32 student_wan_t2v.hidden_size=5120 \
    align_type=downsample upsample_mode=nearest \
    optimizer=AdamW lr=2e-4 weight_decay=0 lr_scheduler=WSDCosineLR ema_decay=null betas=[0.9,0.9] amp=bf16 distributed_method=DDP \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 \
    save_checkpoint_steps=1000 save_samples_steps=null eval_steps=5000 max_steps=20000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1 \
    log=False
```

### Step 1.2: Assemble

```bash
path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1
checkpoint=step_20000.pt
python -m applications.dc_videogen.wan_t2v_assemble \
    patch_size=[1,1,1] input_size=[20,15,26] in_channels=32 \
    depth=40 hidden_size=5120 num_heads=40 ffn_dim=13824 \
    backbone_checkpoint_path=assets/checkpoints/t2v/14b/wan_t2v_14b_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/phase_2_init.pt
```

### Step 2: Patch-Head Fine-Tuning

```bash
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_samples=32 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False vbench_text_prompt.batch_size=1 \
    category_ids=[0] save_samples_at_all_ranks=True \
    mixture.resolution=480F32MS mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=bf16 \
    wan_t2v.depth=40 wan_t2v.hidden_size=5120 wan_t2v.num_heads=40 wan_t2v.ffn_dim=13824 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1/phase_2_init.pt wan_t2v.freeze_backbone=True \
    optimizer=AdamW lr=0.0001 weight_decay=0.0 betas=[0.9,0.999] clip_grad=0.1 warmup_steps=0 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=6000 eval_steps=1000 save_checkpoint_steps=100 save_eval_checkpoint_steps=2000 \
    distributed_method=FSDPWrap \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2 log=False
```

### Step 3: End-to-End Fine-Tuning

```bash
checkpoint=step_6000.pt
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    save_samples_at_all_ranks=True \
    model=wan_t2v model_dtype=bf16 \
    wan_t2v.depth=40 wan_t2v.hidden_size=5120 wan_t2v.num_heads=40 wan_t2v.ffn_dim=13824 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2/$checkpoint \
    wan_t2v.freeze_text_embed=True wan_t2v.freeze_cross_attn=True \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 cfg_scale=5.0 \
    use_lora=True lora_rank=256 lora_alpha=512 \
    optimizer=AdamW lr=0.00005 betas=[0.9,0.999] clip_grad=0.1 weight_decay=0.0 warmup_steps=1000 ema_decay=0.9999 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    distributed_method=FSDPWrap \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=8000 eval_steps=1000 save_checkpoint_steps=100 save_eval_checkpoint_steps=2000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3 log=False

# merge LoRA weights
python -m applications.dc_videogen.convert_lora_into_base \
    load_path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000.pt \
    save_path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000_merged.pt
```

## Model 3: DC-Gen-Wan2.1-I2V-14B

### Step 1.1: Video Embedding Space Alignment

```bash
python -m dc_ai.dc_adapt.trainer mode=train \
    dataset=latent_video data_type=latent model_type=wan_i2v \
    latent_video.data_providers=[LatentFusionXAlign] latent_video.data_ext=.pth \
    latent_video.teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    latent_video.student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    latent_video.batch_size=4 \
    teacher_stae.name=wan-vae \
    student_stae.name=dc-ae-v-1.0-f32t4c32-bf16 student_stae.scaling_factor=0.7241 \
    image_encoder.resolution=480 \
    teacher_wan_i2v.input_size=[20,60,104] teacher_wan_i2v.patch_size=[1,2,2] \
    teacher_wan_i2v.in_channels=16 teacher_wan_i2v.hidden_size=5120 \
    teacher_wan_i2v.pretrained_source=wan teacher_wan_i2v.pretrained_path=assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt \
    student_wan_i2v.input_size=[20,15,26] student_wan_i2v.patch_size=[1,1,1] \
    student_wan_i2v.in_channels=32 student_wan_i2v.hidden_size=5120 \
    align_type=downsample upsample_mode=nearest \
    optimizer=AdamW lr=2e-4 weight_decay=0 lr_scheduler=WSDCosineLR ema_decay=null betas=[0.9,0.9] amp=bf16 distributed_method=DDP \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 \
    save_checkpoint_steps=1000 save_samples_steps=null eval_steps=5000 max_steps=20000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1 \
    log=False
```

### Step 1.2: Assemble

```bash
path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1
checkpoint=step_20000.pt
python -m applications.dc_videogen.wan_i2v_assemble \
    patch_size=[1,1,1] input_size=[20,15,26] in_channels=32 \
    depth=40 hidden_size=5120 num_heads=40 ffn_dim=13824 \
    backbone_checkpoint_path=assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/phase_2_init.pt
```

### Step 2: Patch-Head Fine-Tuning

```bash
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchImagePrompt] \
    vbench_image_prompt.num_samples=32 skip_vbench_evaluator=True category_ids=[0] \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    save_samples_at_all_ranks=True \
    model=wan_i2v model_dtype=bf16 \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_source=dc-ae wan_i2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1/phase_2_init.pt wan_i2v.freeze_backbone=True \
    wan_i2v.train_scheduler=FlowMatchScheduler wan_i2v.flow_shift=5.0 \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 \
    optimizer=AdamW lr=0.0001 weight_decay=0.0 betas=[0.9,0.999] clip_grad=0.1 warmup_steps=0 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=8000 eval_steps=1000 save_checkpoint_steps=100 save_eval_checkpoint_steps=2000 \
    distributed_method=FSDPWrap \
    run_dir=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2 log=False
```

### Step 3: End-to-End Fine-Tuning

```bash
checkpoint=step_8000.pt
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchImagePrompt] \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    save_samples_at_all_ranks=True \
    model=wan_i2v model_dtype=bf16 \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_source=dc-ae wan_i2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2/$checkpoint \
    wan_i2v.freeze_text_embed=True wan_i2v.freeze_cross_attn=True \
    wan_i2v.train_scheduler=FlowMatchScheduler wan_i2v.flow_shift=5.0 \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 cfg_scale=5.0 \
    use_lora=True lora_rank=256 lora_alpha=512 \
    optimizer=AdamW lr=0.00005 betas=[0.9,0.999] clip_grad=0.1 weight_decay=0.0 warmup_steps=1000 ema_decay=0.9999 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    distributed_method=FSDPWrap \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=8000 eval_steps=1000 save_checkpoint_steps=100 save_eval_checkpoint_steps=2000 \
    run_dir=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3 log=False

# merge LoRA weights
python -m applications.dc_videogen.convert_lora_into_base \
    load_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000.pt \
    save_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000_merged.pt
```

## Evaluate DC-Gen Model

Use the merged checkpoints produced at the end of each model's training recipe. Choose manual prompts for a local generation check, or VBench for benchmark generation and scoring.

### Generate from Adapted Checkpoints

After merging LoRA weights, use `manual_prompt` to generate videos without VBench data. These examples use the checkpoints produced by the training recipes above. Both require `pretrained_source=dc-ae`, the 32-channel DC-AE-V, and patch size `[1,1,1]`.

```bash
# DC-Gen-Wan2.1-T2V-1.3B
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=bf16 batch_size=1 \
    prompts="A cat surfing on a sunny day." \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_source=dc-ae \
    wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_40000_merged.pt \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 cfg_scale=3.0 \
    run_dir=exp/demo_videogen/dc_gen_wan2.1_t2v_1.3b

# DC-Gen-Wan2.1-I2V-14B
python -m applications.dc_videogen.manual_prompt \
    model=wan_i2v resolution=480 amp=bf16 model_dtype=bf16 offload=True batch_size=1 \
    'image_paths=[path/to/input.jpg]' prompts="The subject moves gently while the camera slowly pans right." \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] wan_i2v.text_encoder_id=wan2.1-i2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_source=dc-ae \
    wan_i2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000_merged.pt \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 wan_i2v.flow_shift=5.0 cfg_scale=5.0 \
    run_dir=exp/demo_videogen/dc_gen_wan2.1_i2v_14b
```

For an adapted T2V-14B checkpoint, use the T2V command with `wan_t2v.depth=40 wan_t2v.hidden_size=5120 wan_t2v.num_heads=40 wan_t2v.ffn_dim=13824 offload=True cfg_scale=5.0`, set `wan_t2v.pretrained_path` to the T2V-14B `phase_3/step_8000_merged.pt` below, and choose a separate `run_dir`.

### Run VBench Inference

Complete [Prepare Evaluation Benchmark](#prepare-evaluation-benchmark) first. Use `skip_vbench_evaluator=True` to only generate videos, or `False` to also run VBench scoring. Set `vbench_text_prompt.num_videos_per_prompt=N` for T2V or `vbench_image_prompt.num_videos_per_prompt=N` for I2V to generate N videos per prompt.

#### DC-Gen-Wan2.1-T2V-1.3B

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=eval \
    eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_videos_per_prompt=5 \
    vbench_text_prompt.batch_size=1 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False \
    category_ids=[0,1,2,3,4,5,6,7,8,9,10] save_samples_at_all_ranks=True \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=bf16 amp=bf16 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_40000_merged.pt \
    wan_t2v.pretrained_source=dc-ae \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 cfg_scale=3.0 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/eval_vbench
```

#### DC-Gen-Wan2.1-T2V-14B

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=eval \
    eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_videos_per_prompt=5 \
    vbench_text_prompt.batch_size=1 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False \
    category_ids=[0,1,2,3,4,5,6,7,8,9,10] save_samples_at_all_ranks=True \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=bf16 amp=bf16 offload=True \
    wan_t2v.depth=40 wan_t2v.hidden_size=5120 wan_t2v.num_heads=40 wan_t2v.ffn_dim=13824 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000_merged.pt \
    wan_t2v.pretrained_source=dc-ae \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 cfg_scale=5.0 \
    run_dir=exp/dc_gen/fusionX/wan2.1_t2v_14B_to_dc_ae_v_f32t4c32_1.0/eval_vbench
```

#### DC-Gen-Wan2.1-I2V-14B

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=eval \
    eval_data_providers=[VBenchImagePrompt] \
    vbench_image_prompt.category_id=3 vbench_image_prompt.num_videos_per_prompt=5 \
    vbench_image_prompt.batch_size=1 vbench_image_prompt.num_workers=0 vbench_image_prompt.persistent_workers=False \
    vbench_image_prompt.image_folder=assets/data/vbench/vbench_i2v_imgs/832-480 \
    category_ids=[0,1,2] save_samples_at_all_ranks=True \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_i2v model_dtype=bf16 amp=bf16 offload=True \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_path=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_8000_merged.pt \
    wan_i2v.pretrained_source=dc-ae \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 wan_i2v.flow_shift=5.0 cfg_scale=5.0 \
    run_dir=exp/dc_gen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/eval_vbench
```
