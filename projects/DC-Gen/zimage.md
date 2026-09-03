# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

This README file contains guidelines for training ZImage models with DC-Gen on 512px and 1K.

> **Note**: The parameters in this guide are for debugging purposes only. For optimal training hyperparameters, please refer to Appendix E of the [DC-Gen Paper](https://arxiv.org/abs/2509.25180).




## Convert Checkpoint

```bash
python -m applications.dc_gen.convert_checkpoint \
    model=z_image_turbo \
    save_path=assets/checkpoints/t2i/zimage_turbo_pretrained.pt
```

## Prepare Null Embedding

```bash
python -m applications.dc_gen.get_null_embedding \
    id=z-image-turbo/qwen3-4b-512-bf16
```

## Test Pre-Trained Model

```bash
python -m applications.dc_gen.manual_prompt \
    model=zimage \
    cfg_scale=0.0 \
    resolution=1024 \
    amp=bf16 \
    model_dtype=bf16 \
    autoencoder_dtype=fp16 \
    zimage.in_channels=16 \
    zimage.patch_size=2 \
    zimage.all_patch_size=[2] \
    zimage.all_f_patch_size=[1] \
    zimage.eval_scheduler=FluxScheduler \
    zimage.num_inference_steps=8 \
    zimage.flow_shift=3.0 \
    zimage.pretrained_source=diffusers \
    zimage.pretrained_path=assets/checkpoints/t2i/zimage_turbo_pretrained.pt \
    autoencoder.name=flux-vae \
    text_encoders=[z-image-turbo/qwen3-4b-512-bf16] \
    prompts=["A cute cat playing with a ball"] \
    run_dir=tmp_zimage_turbo_inference \
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

Please download the example training data from https://drive.google.com/drive/folders/1koHmuXg6GYKvDLcc_PQ_HupQcSEM6od8 to an appropriate path.

### Step 1: Extract Meta for RGB Tar Files

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/rgb/tar/shards> \
    save_path=<path/to/save/rgb_meta.json>

```

### Step 2: Extract Latent

```bash
# 512px flux-vae (teacher for Stage 1)
python -m dc_ai.t2icore.data_provider.multiprocess_generate_latent_archives \
    meta_path=<path/to/rgb_meta.json> \
    original_archive_dir=<path/to/rgb/tar/shards> \
    save_dir=<path/to/latent/flux_vae_512> \
    resolution=512 batch_size=4 \
    autoencoder.name=flux-vae \
    run_dir=tmp_extract_latent_flux_vae_512

# 512px dc-ae (student for Stage 1)
python -m dc_ai.t2icore.data_provider.multiprocess_generate_latent_archives \
    meta_path=<path/to/rgb_meta.json> \
    original_archive_dir=<path/to/rgb/tar/shards> \
    save_dir=<path/to/latent/dc_ae_f32c32_512> \
    resolution=512 batch_size=4 \
    autoencoder.name=dc-ae-f32c32-in-1.0 \
    autoencoder.scaling_factor=0.3189 \
    run_dir=tmp_extract_latent_dc_ae_512

# 1024px dc-ae (for Stage 3)
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
    save_path=assets/data/meta/wids/flux_vae_512/<dataset>/zimage_flux_reason.json

python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/latent/dc_ae_f32c32_512> \
    save_path=assets/data/meta/wids/dc_ae_f32c32_in_1.0_512/<dataset>/zimage_flux_reason.json

python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=<path/to/latent/dc_ae_f32c32_1024> \
    save_path=assets/data/meta/wids/dc_ae_f32c32_in_1.0_1024/<dataset>/zimage_flux_reason.json
```

## Step 1: DC-Gen (f8 -> f32) on 512px Generation

### Stage 1.1: Align Embedding

```bash
torchrun --nnodes=1 --nproc_per_node=4 -m dc_ai.dc_adapt.trainer mode=train \
    log=True \
    lr=1e-4 ema_decay=null betas=[0.9,0.95] \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 max_steps=50000 eval_steps=5000 \
    distributed_method=FSDPWrap amp=bf16 \
    dataset=latent_image data_type=latent \
    latent_image.resolution=512 \
    latent_image.teacher_wds_meta_dir=assets/data/meta/wids/flux_vae_512/zimage \
    latent_image.student_wds_meta_dir=assets/data/meta/wids/dc_ae_f32c32_in_1.0_512/zimage \
    save_checkpoint_steps=1000 \
    latent_image.save_checkpoint_steps=1000 \
    latent_image.batch_size=8 \
    latent_image.data_providers=[LatentZImageFluxReasonAlign] \
    save_samples_steps=null \
    teacher_ae.name=flux-vae \
    student_ae.name=dc-ae-f32c32-in-1.0-diffusers \
    model_type=zimage \
    teacher_zimage.input_size=64 teacher_zimage.patch_size=2 teacher_zimage.in_channels=16 \
    teacher_zimage.pretrained_source=diffusers \
    teacher_zimage.pretrained_path=assets/checkpoints/t2i/zimage_turbo_pretrained.pt \
    student_zimage.input_size=16 student_zimage.patch_size=1 student_zimage.in_channels=32 \
    align_type=upsample upsample_mode=bilinear \
    run_dir=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_1_512_0223_test \
    timeout=3600
```

### Stage 1.2: Assemble

```bash
path=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_1_512_0223_test
checkpoint=step_50000.pt
python -m applications.dc_gen.zimage_assemble \
    patch_size=1 input_size=16 in_channels=32 \
    all_patch_size=[1] all_f_patch_size=[1] \
    backbone_checkpoint_path=assets/checkpoints/t2i/zimage_turbo_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/zimage_turbo_dc_ae_init_test.pt
```

### Stage 2: Align Patch-Head

```bash
path=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_1_512_0223_test/zimage_turbo_dc_ae_init_test.pt
torchrun --nnodes=1 --nproc_per_node=4 -m dc_ai.t2icore.trainer mode=train \
    resolution=512 mixture.resolution=512 \
    mixture.data_providers=[LatentZImageFluxReason] \
    mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_f32c32_in_1.0_512 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0-diffusers autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=zimage zimage.in_channels=32 zimage.patch_size=1 \
    zimage.all_patch_size=[1] zimage.all_f_patch_size=[1] \
    zimage.use_guide_loss=True zimage.train_cfg_value=3.0 \
    zimage.drop_text_ratio=0.1 \
    zimage.base_seq_len=4096 \
    zimage.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    zimage.eval_scheduler=Flux2Scheduler zimage.num_inference_steps=9 \
    zimage.pretrained_source=dc-gen zimage.pretrained_path=$path \
    zimage.freeze_backbone=True \
    activation_checkpointing_mode=transformer \
    zimage.null_embeds_dir=assets/data/null_text_embeddings \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=512 mjhq_text_prompt.num_samples=2000 \
    mjhq_text_prompt.batch_size=16 cfg_scale=0.0 \
    text_encoders=[z-image-turbo/qwen3-4b-512-bf16] \
    warmup_steps=1000 max_steps=20000 \
    save_checkpoint_steps=200 eval_steps=1000 \
    save_eval_checkpoint_steps=1000 \
    save_image_format=jpeg amp=bf16 lr=1e-4 clip_grad=0.1 \
    ema_decay=null timeout=3600 \
    run_dir=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_2_512_flux2_cfg_3_0223_test \
    log=True
```

### Stage 3: End-to-End Fine-Tuning on 512px

```bash
path=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_2_512_flux2_cfg_3_0223_test/step_200.pt
torchrun --nnodes=1 --nproc_per_node=4 -m dc_ai.t2icore.trainer mode=train \
    resolution=512 mixture.resolution=512 \
    mixture.data_providers=[LatentZImageFluxReason] \
    mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_f32c32_in_1.0_512 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0-diffusers autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=zimage zimage.in_channels=32 zimage.patch_size=1 \
    zimage.all_patch_size=[1] zimage.all_f_patch_size=[1] \
    zimage.use_guide_loss=True zimage.train_cfg_value=3.0 \
    zimage.drop_text_ratio=0.1 \
    zimage.base_seq_len=4096 \
    zimage.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    zimage.eval_scheduler=Flux2Scheduler zimage.num_inference_steps=9 \
    zimage.use_lora=True zimage.lora_rank=256 zimage.lora_alpha=256 \
    zimage.null_embeds_dir=assets/data/null_text_embeddings \
    zimage.pretrained_source=dc-gen zimage.pretrained_path=$path zimage.load_before_lora=True \
    activation_checkpointing_mode=transformer \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=512 mjhq_text_prompt.num_samples=2000 \
    mjhq_text_prompt.batch_size=16 cfg_scale=0.0 \
    text_encoders=[z-image-turbo/qwen3-4b-512-bf16] \
    warmup_steps=1000 max_steps=10000 \
    save_checkpoint_steps=200 eval_steps=1000 \
    save_eval_checkpoint_steps=1000 \
    amp=bf16 lr=1e-4 clip_grad=0.1 betas=[0.9,0.95] weight_decay=0.0 \
    ema_decay=0.999 ema_warmup_steps=0 timeout=3600 \
    save_image_format=jpeg \
    run_dir=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_3_512_flux2_cfg_3_0223_test \
    log=True
```

## Step 2: Lift Resolution from 512px to 1024px

### Stage 2.1: Merge LoRA

```bash
python -m applications.dc_gen.convert_lora_into_base \
    load_path=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_3_512_flux2_cfg_3_0223_test/step_10000.pt \
    save_path=assets/checkpoints/t2i/zimage_512px_dcae_1_0_f32c32_0223_cfg_3_flux2_test_step_10000.pt
```

### Stage 2.2: End-to-End Fine-Tuning on 1024px

```bash
path=assets/checkpoints/t2i/zimage_512px_dcae_1_0_f32c32_0223_cfg_3_flux2_test_step_10000.pt
torchrun --nnodes=1 --nproc_per_node=4 -m dc_ai.t2icore.trainer mode=train \
    resolution=1024 mixture.resolution=1024 \
    mixture.data_providers=[LatentZImageFluxReason] \
    mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_f32c32_in_1.0_1024 \
    mixture.batch_size=64 \
    autoencoder.name=dc-ae-f32c32-in-1.0-diffusers autoencoder.scaling_factor=0.3189 \
    distributed_method=FSDPWrap \
    model=zimage zimage.in_channels=32 zimage.patch_size=1 \
    zimage.all_patch_size=[1] zimage.all_f_patch_size=[1] \
    zimage.use_guide_loss=True zimage.train_cfg_value=3.0 \
    zimage.drop_text_ratio=0.1 \
    zimage.base_seq_len=4096 \
    zimage.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    zimage.eval_scheduler=Flux2Scheduler zimage.num_inference_steps=9 \
    zimage.use_lora=True zimage.lora_rank=256 zimage.lora_alpha=256 \
    zimage.null_embeds_dir=assets/data/null_text_embeddings \
    zimage.pretrained_source=dc-gen zimage.pretrained_path=$path zimage.load_before_lora=True \
    activation_checkpointing_mode=transformer \
    eval_data_providers=[MJHQTextPrompt] \
    mjhq_text_prompt.resolution=1024 mjhq_text_prompt.num_samples=2000 \
    mjhq_text_prompt.batch_size=16 cfg_scale=0.0 \
    text_encoders=[z-image-turbo/qwen3-4b-512-bf16] \
    warmup_steps=1000 max_steps=10000 \
    save_checkpoint_steps=100 eval_steps=1000 \
    save_eval_checkpoint_steps=1000 \
    amp=bf16 lr=1e-4 clip_grad=0.1 betas=[0.9,0.95] weight_decay=0.0 \
    ema_decay=0.999 ema_warmup_steps=0 timeout=3600 \
    save_image_format=jpeg \
    run_dir=exp/dc_gen/ZImage/zimage_turbo_6B_to_dc_ae_f32c32_1.0/phase_3_1024_flux2_cfg_3_0223_test \
    log=True
```
