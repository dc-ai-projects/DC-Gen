# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

This README file contains guidelines for training FLUX models with DC-Gen on 512px and 1K.

> **Note**: The parameters in this guide are for debugging purposes only. For optimal training hyperparameters, please refer to Appendix E of the [DC-Gen Paper](https://arxiv.org/abs/2509.25180).




## Convert Checkpoint

```bash
python -m applications.dc_gen.convert_checkpoint model=flux_krea save_path=assets/checkpoints/t2i/flux_krea_pretrained.pt
```

## Prepare Null Embedding

> **Note:** Requires access to `black-forest-labs/FLUX.1-Krea-dev` on HuggingFace. Request access at https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev and authenticate with `huggingface-cli login` before running.

```bash
python -m applications.dc_gen.get_null_embedding id=flux.1-krea-dev/t5-256-bf16
python -m applications.dc_gen.get_null_embedding id=flux.1-krea-dev/clip-77-bf16
```

## Test Pre-Trained Model

```bash
python -m applications.dc_gen.manual_prompt \
    model=flux \
    cfg_scale=3.5 \
    resolution=1024 \
    amp=fp16 \
    model_dtype=fp16 \
    autoencoder_dtype=fp16 \
    flux.pretrained_source=flux \
    flux.pretrained_path=assets/checkpoints/t2i/flux_krea_pretrained.pt \
    flux.clip_text_encoder_id=flux.1-krea-dev/clip-77-bf16 \
    flux.t5_text_encoder_id=flux.1-krea-dev/t5-256-bf16 \
    autoencoder.name=flux-vae \
    text_encoders=[flux.1-krea-dev/clip-77-bf16,flux.1-krea-dev/t5-256-bf16] \
    'prompts=["A cute cat playing with a ball"]' \
    run_dir=tmp_flux_krea_inference \
    seed=42
```

## Prepare Evaluation Benchmark

```bash
hf download --repo-type dataset playgroundai/MJHQ-30K \
    mjhq30k_imgs.zip meta_data.json \
    --local-dir ~/dataset/MJHQ-30K
cd ~/dataset/MJHQ-30K && unzip mjhq30k_imgs.zip -d imgs
cd -

torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.apps.metrics.fid.compute_fid \
    resolution=512 suffix=".jpg" \
    data_dir=~/dataset/MJHQ-30K/imgs \
    fid.save_path=~/dataset/MJHQ-30K/MJHQ_30K_512px_fid_embeddings_30000.npz
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.apps.metrics.fid.compute_fid \
    resolution=1024 suffix=".jpg" \
    data_dir=~/dataset/MJHQ-30K/imgs \
    fid.save_path=~/dataset/MJHQ-30K/MJHQ_30K_1024px_fid_embeddings_30000.npz
```

## Prepare Training Data

Please download the example training data from https://drive.google.com/drive/folders/1FjBivCurr7gQ7GmBH5rIAxwhd0X9dzVb to an appropriate path.

### Step 1: Extract Meta for RGB Tar Files

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/rgb/tar/shards> \
    save_path=<path/to/save/rgb_meta.json>
```

### Step 2: Extract Latent

```bash
# 512px flux-vae
python -m dc_ai.t2icore.data_provider.multiprocess_generate_latent_archives \
    meta_path=<path/to/rgb_meta.json> \
    original_archive_dir=<path/to/rgb/tar/shards> \
    save_dir=<path/to/latent/flux_vae_512> \
    resolution=512 batch_size=4 \
    autoencoder.name=flux-vae \
    run_dir=tmp_extract_latent_flux_vae_512

# 512px dc-ae
python -m dc_ai.t2icore.data_provider.multiprocess_generate_latent_archives \
    meta_path=<path/to/rgb_meta.json> \
    original_archive_dir=<path/to/rgb/tar/shards> \
    save_dir=<path/to/latent/dc_ae_f32c32_512> \
    resolution=512 batch_size=4 \
    autoencoder.name=dc-ae-f32c32-in-1.0 \
    autoencoder.scaling_factor=0.3189 \
    run_dir=tmp_extract_latent_dc_ae_512

# 1024px dc-ae
python -m dc_ai.t2icore.data_provider.multiprocess_generate_latent_archives \
    meta_path=<path/to/rgb_meta.json> \
    original_archive_dir=<path/to/rgb/tar/shards> \
    save_dir=<path/to/latent/dc_ae_f32c32_1024> \
    resolution=1024 batch_size=4 \
    autoencoder.name=dc-ae-f32c32-in-1.0 \
    autoencoder.scaling_factor=0.3189 \
    run_dir=tmp_extract_latent_dc_ae_1024
```

### Step 3: Generate Meta for Latent Tar Files

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/latent/flux_vae_512> \
    save_path=assets/data/meta/latents_512/flux_vae/igdata.json

python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/latent/dc_ae_f32c32_512> \
    save_path=assets/data/meta/latents_512/dc-ae-f32c32-in-1.0/igdata.json

python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/latent/dc_ae_f32c32_1024> \
    save_path=assets/data/meta/latents_1024/dc-ae-f32c32-in-1.0/igdata.json
```

## Step 1: DC-Gen (f8 -> f32) on 512*512 Generation

### Stage 1.1: Align Embedding 

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.dc_adapt.trainer mode=train \
    log=True \
    lr=1e-4 ema_decay=null betas=[0.9,0.95] \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 max_steps=20000 \
    distributed_method=FSDPWrap \
    dataset=latent_image data_type=latent \
    latent_image.resolution=512 \
    latent_image.teacher_wds_meta_dir=assets/data/meta/latents_512/flux_vae \
    latent_image.student_wds_meta_dir=assets/data/meta/latents_512/dc-ae-f32c32-in-1.0 \
    save_checkpoint_steps=1000 \
    latent_image.save_checkpoint_steps=1000 \
    latent_image.batch_size=8 \
    latent_image.data_providers=[LatentIGAlign] \
    save_samples_steps=null \
    teacher_ae.name=flux-vae \
    student_ae.name=dc-ae-f32c32-in-1.0 \
    model_type=flux \
    teacher_flux.input_size=64 teacher_flux.patch_size=2 teacher_flux.in_channels=16 \
    teacher_flux.pretrained_source=flux \
    teacher_flux.pretrained_path=assets/checkpoints/t2i/flux_krea_pretrained.pt \
    student_flux.input_size=16 student_flux.patch_size=1 student_flux.in_channels=32 \
    align_type=upsample upsample_mode=bilinear \
    run_dir=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_1_512_test
```

### Stage 1.2: Assemble

```bash
path=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_1_512_test
checkpoint=step_20000.pt
python -m applications.dc_gen.flux_assemble \
    patch_size=1 input_size=16 in_channels=32 \
    backbone_checkpoint_path=assets/checkpoints/t2i/flux_krea_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/flux_f32_krea_init.pt
```

### Stage 2: Align Patch-Head 

```bash
path=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_1_512_test/flux_f32_krea_init.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.t2icore.trainer mode=train \
    resolution=512 mixture.resolution=512 \
    mixture.data_providers=[LatentIG] \
    mixture.wds_meta_dir=assets/data/meta/latents_512/dc-ae-f32c32-in-1.0 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=flux flux.drop_text_raio=0.1 flux.in_channels=32 flux.patch_size=1 \
    flux.use_guide_loss=True \
    flux.train_scheduler=FlowMatchEulerDiscreteScheduler \
    flux.eval_scheduler=FluxScheduler flux.num_inference_steps=20 \
    flux.pretrained_source=flux flux.pretrained_path=$path \
    flux.freeze_backbone=True \
    activation_checkpointing_mode=transformer \
    flux.null_embeds_dir=assets/data/null_text_embeddings \
    flux.clip_text_encoder_id=flux.1-krea-dev/clip-77-bf16 \
    flux.t5_text_encoder_id=flux.1-krea-dev/t5-256-bf16 \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=512 mjhq_text_prompt.num_samples=2000 \
    cfg_scale=3.5 \
    text_encoders=[flux.1-krea-dev/clip-77-bf16,flux.1-krea-dev/t5-256-bf16] \
    warmup_steps=1000 max_steps=5000 \
    save_checkpoint_steps=1000 eval_steps=2000 \
    save_eval_checkpoint_steps=2000 save_image_format=jpeg \
    amp=bf16 lr=1e-4 clip_grad=0.1 ema_decay=null \
    run_dir=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_2_512_test \
    log=True
```


### Stage 3: End-to-End Fine-Tuning on 512px

```bash
path=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_2_512_test/step_5000.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.t2icore.trainer mode=train \
    resolution=512 mixture.resolution=512 \
    mixture.data_providers=[LatentIG] \
    mixture.wds_meta_dir=assets/data/meta/latents_512/dc-ae-f32c32-in-1.0 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=flux flux.drop_text_raio=0.1 flux.in_channels=32 flux.patch_size=1 \
    flux.use_guide_loss=True \
    flux.train_scheduler=FlowMatchEulerDiscreteScheduler \
    flux.eval_scheduler=FluxScheduler flux.num_inference_steps=20 \
    flux.use_lora=True flux.lora_rank=256 flux.lora_alpha=256 \
    flux.null_embeds_dir=assets/data/null_text_embeddings \
    flux.pretrained_source=dc-gen flux.pretrained_path=$path \
    flux.load_before_lora=True \
    activation_checkpointing_mode=transformer \
    flux.clip_text_encoder_id=flux.1-krea-dev/clip-77-bf16 \
    flux.t5_text_encoder_id=flux.1-krea-dev/t5-256-bf16 \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=512 mjhq_text_prompt.num_samples=2000 \
    cfg_scale=3.5 \
    text_encoders=[flux.1-krea-dev/clip-77-bf16,flux.1-krea-dev/t5-256-bf16] \
    warmup_steps=1000 max_steps=10000 \
    save_checkpoint_steps=1000 eval_steps=2000 \
    save_eval_checkpoint_steps=2000 save_image_format=jpeg \
    amp=bf16 lr=1e-4 clip_grad=0.1 betas=[0.9,0.95] weight_decay=0.0 \
    ema_decay=0.999 ema_warmup_steps=0 \
    run_dir=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_3_512_test \
    log=True
```

## Step 2: Lift Resolution from 512px to 1024px

### Stage 2.1: Merge LoRA

```bash
python -m applications.dc_gen.convert_lora_into_base \
    load_path=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_3_512_test/step_10000.pt \
    save_path=assets/checkpoints/t2i/flux_krea_512px_dcae_f32c32_1.0.pt
```

### Stage 2.2: End-to-End Fine-Tuning on 1024px

```bash
path=assets/checkpoints/t2i/flux_krea_512px_dcae_f32c32_1.0.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.t2icore.trainer mode=train \
    resolution=1024 mixture.resolution=1024 \
    mixture.data_providers=[LatentIG] \
    mixture.wds_meta_dir=assets/data/meta/latents_1024/dc-ae-f32c32-in-1.0 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=flux flux.drop_text_raio=0.1 flux.in_channels=32 flux.patch_size=1 \
    flux.use_guide_loss=True \
    flux.train_scheduler=FlowMatchEulerDiscreteScheduler \
    flux.eval_scheduler=FluxScheduler flux.num_inference_steps=20 \
    flux.use_lora=True flux.lora_rank=256 flux.lora_alpha=256 \
    flux.null_embeds_dir=assets/data/null_text_embeddings \
    flux.pretrained_source=dc-gen flux.pretrained_path=$path \
    flux.load_before_lora=True \
    activation_checkpointing_mode=transformer \
    flux.clip_text_encoder_id=flux.1-krea-dev/clip-77-bf16 \
    flux.t5_text_encoder_id=flux.1-krea-dev/t5-256-bf16 \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=1024 mjhq_text_prompt.num_samples=2000 \
    cfg_scale=3.5 \
    text_encoders=[flux.1-krea-dev/clip-77-bf16,flux.1-krea-dev/t5-256-bf16] \
    warmup_steps=1000 max_steps=100000 \
    save_checkpoint_steps=1000 eval_steps=2000 \
    save_eval_checkpoint_steps=2000 save_image_format=jpeg \
    amp=bf16 lr=1e-4 clip_grad=0.1 betas=[0.9,0.95] weight_decay=0.0 \
    ema_decay=0.999 ema_warmup_steps=0 \
    run_dir=exp/dc_gen/Flux/flux_krea_to_dc_ae_f32c32_1.0/phase_4_1024_test \
    log=True
```
