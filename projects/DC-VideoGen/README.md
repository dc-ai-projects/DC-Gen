# DC-VideoGen

Please follow the environment setup instructions in [README.md](../../README.md).

## DC-AE-V

See [DC-AE-V.md](./DC-AE-V.md) for using and evaluating DC-AE-V models.

## AE-Adapt-V

### Environment

``` bash
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
pip install git+https://github.com/WenkunHe/VBench.git --no-build-isolation
pip install numpy==1.26.4 easydict fairscale
```

### Prepare Text Encoder

``` bash
hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir assets/checkpoints/t2v
python -m applications.dc_videogen.get_null_embedding \
    id=wan2.1-t2v/umt5-512-bf16
python -m applications.dc_videogen.get_null_embedding \
    id=wan2.1-i2v/umt5-512-bf16
```

You should be able to see `assets/checkpoints/t2v/models_t5_umt5-xxl-enc-bf16.pth`, `assets/data/null_text_embeddings/wan2.1-t2v/umt5-512-bf16.pth`, and `assets/data/null_text_embeddings/wan2.1-i2v/umt5-512-bf16.pth`.

### Prepare Image Encoder

``` bash
hf download Wan-AI/Wan2.1-I2V-14B-480P models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth --local-dir assets/checkpoints/i2v
```

You should be able to see `assets/checkpoints/i2v/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`.

### Prepare Evaluation Data

#### T2V

``` bash
mkdir -p assets/data/vbench
wget https://raw.githubusercontent.com/Vchitect/VBench/refs/heads/master/vbench/VBench_full_info.json -O assets/data/vbench/VBench_full_info.json
wget https://raw.githubusercontent.com/Vchitect/VBench/refs/heads/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt -O assets/data/vbench/vbench_extended.txt
python -m applications.dc_videogen.rewrite_vbench_meta
```

You should be able to see `assets/data/vbench/VBench_extended_full_info.json`.

#### I2V

Please download `VBench_i2v_extended_full_info.json` from https://drive.google.com/drive/folders/1kyUG_t9r1IPr6hmb0UG9igGwy54nadph to `assets/data/vbench/VBench_i2v_extended_full_info.json`, and `vbench_i2v_imgs` to `assets/data/vbench/vbench_i2v_imgs`.

You should be able to see `assets/data/vbench/VBench_i2v_extended_full_info.json` and 355 images for each subfolder under `assets/data/vbench/vbench_i2v_imgs`.

### Evaluate Pre-Trained DC-VideoGen-Wan Models (TODO)

See [DC-VideoGen-Eval.md](./DC-VideoGen-Eval.md).

### Training Pipeline of AE-Adapt-V

See [AE-Adapt-V.md](./AE-Adapt-V.md).
