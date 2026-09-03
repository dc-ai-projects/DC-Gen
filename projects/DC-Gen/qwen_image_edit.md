# DC-Gen: Image Editing with DC-Gen Based on Qwen-Image-Edit

This README file contains guidelines for training Qwen-Image-Edit models with DC-Gen on 512px and 1K.

> **Note**: The parameters in this guide are for debugging purposes only. For optimal training hyperparameters, please refer to Appendix E of the [DC-Gen Paper](https://arxiv.org/abs/2509.25180).




## Convert Checkpoint

```bash
python -m applications.image_edit.convert_checkpoint \
    model=qwen_image \
    save_path=assets/checkpoints/image_edit/qwen_image_pretrained.pt
```

## Test Pre-Trained Model

```bash
python -m applications.image_edit.manual_prompt \
    model=qwen_image resolution=1024 amp=bf16 model_dtype=bf16 \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=1024 \
    autoencoder.name=qwen-vae autoencoder_dtype=bf16 \
    qwen_image.in_channels=16 qwen_image.depth=60 qwen_image.hidden_dim=3072 qwen_image.num_heads=24 \
    qwen_image.pretrained_path=assets/checkpoints/image_edit/qwen_image_pretrained.pt qwen_image.pretrained_source=qwen_image \
    cfg_scale=4.0 \
    run_dir=tmp_pretrained_qwen_image
```

## Prepare Evaluation Benchmark

```bash
pip install datasets
hf download stepfun-ai/GEdit-Bench --local-dir assets/data/GEdit_raw --repo-type dataset
python -m applications.image_edit.gedit_processing input_dir=assets/data/GEdit_raw output_dir=assets/data/GEdit
rm -rf assets/data/GEdit_raw
```

## Evaluate Pre-Trained Model

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.trainer mode=eval \
    eval_data_providers=[GEdit] gedit.batch_size=1 gedit.resolution=1024 \
    model=qwen_image resolution=1024 amp=bf16 model_dtype=bf16 \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=1024 \
    autoencoder.name=qwen-vae autoencoder_dtype=bf16 \
    qwen_image.in_channels=16 qwen_image.depth=60 qwen_image.hidden_dim=3072 qwen_image.num_heads=24 \
    qwen_image.pretrained_path=assets/checkpoints/image_edit/qwen_image_pretrained.pt qwen_image.pretrained_source=qwen_image \
    cfg_scale=4.0 \
    save_samples_at_all_ranks=True num_save_samples=1212 save_input_images=True \
    run_dir=tmp_pretrained_qwen_image
```

## Prepare Training Data

Please download the example of the processed Pico-Banana data (`single_turn_processed`) and the Qwen-Image-Edit generated data (`qwen_image_gen`) from https://drive.google.com/drive/folders/1kV8r8EH1rdmy2VNrkqZmYt3X4wWvzGBZ to an appropriate path.

```bash
PICO_BANANA_DATA_PATH=<path_to_pico_banana_data>
```

### Step 1: Generate Meta for RGB Data

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=$PICO_BANANA_DATA_PATH/single_turn_processed \
    save_path=assets/data/meta/pico_banana_raw.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=$PICO_BANANA_DATA_PATH/qwen_image_gen \
    save_path=assets/data/meta/pico_banana_qwen_image_gen.json
```

### Step 2: Extract Latent

```bash
# Extract latent (single process; adjust task_id / num_archives_per_task for parallelism)
python -m dc_ai.imageeditcore.data_provider.generate_latent_archives \
    task_id=0 \
    dataset_name=pico_banana resolution=512F32MS \
    latent_ext=.pth \
    meta_path=assets/data/meta/pico_banana_qwen_image_gen.json \
    original_archive_dir=$PICO_BANANA_DATA_PATH/qwen_image_gen \
    autoencoder.name=qwen-vae \
    size_transform=QwenImageResize \
    batch_size=16 num_archives_per_task=10 \
    save_dir=assets/dataset/latent/qwen_image_gen/pico_banana/qwen_vae_512

python -m dc_ai.imageeditcore.data_provider.generate_latent_archives \
    task_id=0 \
    dataset_name=pico_banana resolution=512F32MS \
    latent_ext=.pth \
    meta_path=assets/data/meta/pico_banana_qwen_image_gen.json \
    original_archive_dir=$PICO_BANANA_DATA_PATH/qwen_image_gen \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    size_transform=QwenImageResize \
    batch_size=16 num_archives_per_task=10 \
    save_dir=assets/dataset/latent/qwen_image_gen/pico_banana/dc_ae_f32c32_in_1.0_512

python -m dc_ai.imageeditcore.data_provider.generate_latent_archives \
    task_id=0 \
    dataset_name=pico_banana resolution=512F32MS \
    latent_ext=.pth \
    meta_path=assets/data/meta/pico_banana_raw.json \
    original_archive_dir=$PICO_BANANA_DATA_PATH/single_turn_processed \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    size_transform=QwenImageResize \
    batch_size=16 num_archives_per_task=10 \
    save_dir=assets/dataset/latent/pico_banana/dc_ae_f32c32_in_1.0_512

python -m dc_ai.imageeditcore.data_provider.generate_latent_archives \
    task_id=0 \
    dataset_name=pico_banana resolution=1024 \
    latent_ext=.pth \
    meta_path=assets/data/meta/pico_banana_raw.json \
    original_archive_dir=$PICO_BANANA_DATA_PATH/single_turn_processed \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    size_transform=QwenImageResize \
    batch_size=16 num_archives_per_task=10 \
    save_dir=assets/dataset/latent/pico_banana/dc_ae_f32c32_in_1.0_1024
```

### Step 3: Generate Meta for Latent

```bash
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=assets/dataset/latent/qwen_image_gen/pico_banana/qwen_vae_512 \
    save_path=assets/data/meta/qwen_vae_512/pico_banana_qwen_image_gen.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=assets/dataset/latent/qwen_image_gen/pico_banana/dc_ae_f32c32_in_1.0_512 \
    save_path=assets/data/meta/dc_ae_f32c32_in_1.0_512/pico_banana_qwen_image_gen.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=assets/dataset/latent/pico_banana/dc_ae_f32c32_in_1.0_512 \
    save_path=assets/data/meta/dc_ae_f32c32_in_1.0_512/pico_banana_single.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta \
    data_dir=assets/dataset/latent/pico_banana/dc_ae_f32c32_in_1.0_1024 \
    save_path=assets/data/meta/dc_ae_f32c32_in_1.0_1024/pico_banana_single.json
```

## Step 1: DC-Gen on 512px

### Stage 1.1: Align Patch Embedding

```bash
python -m dc_ai.dc_adapt.trainer mode=train \
    log=False \
    distributed_method=FSDPWrap \
    dataset=latent_image_edit data_type=latent \
    latent_image_edit.data_ext=".pth" latent_image_edit.resolution=512F32MS \
    latent_image_edit.data_providers=[LatentPicoBananaQwenImageGenAlign] \
    latent_image_edit.teacher_wds_meta_dir=assets/data/meta/qwen_vae_512 \
    latent_image_edit.student_wds_meta_dir=assets/data/meta/dc_ae_f32c32_in_1.0_512 \
    latent_image_edit.batch_size=64 \
    teacher_ae.name=qwen-vae student_ae.name=dc-ae-f32c32-in-1.0 \
    model_type=qwen_image \
    teacher_qwen_image.input_size=64 teacher_qwen_image.patch_size=2 teacher_qwen_image.in_channels=16 \
    teacher_qwen_image.pretrained_source=qwen_image \
    teacher_qwen_image.pretrained_path=assets/checkpoints/image_edit/tuned_qwen_image_512px.pt \
    student_qwen_image.input_size=16 student_qwen_image.patch_size=1 student_qwen_image.in_channels=32 \
    align_type=upsample upsample_mode=bilinear \
    amp=bf16 lr=1e-4 ema_decay=null betas=[0.9,0.95] \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 max_steps=50000 eval_steps=5000 \
    save_checkpoint_steps=1000 \
    run_dir=exp/qwen_vae/pico_banana_qwen_image_gen/align_patch_embedding
```

### Stage 1.2: Assemble

```bash
path=exp/qwen_vae/pico_banana_qwen_image_gen/align_patch_embedding
checkpoint=step_50000.pt
python -m applications.image_edit.qwen_image_assemble \
    patch_size=1 input_size=16 in_channels=32 \
    backbone_checkpoint_path=assets/checkpoints/image_edit/tuned_qwen_image_512px.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/qwen_image_init.pt
```

### Stage 2: Patch-Head Fine-Tuning

```bash
path=exp/qwen_vae/pico_banana_qwen_image_gen/align_patch_embedding/qwen_image_init.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.trainer mode=train \
    log=False \
    resolution=512F32MS amp=bf16 model_dtype=bf16 \
    distributed_method=FSDPWrap \
    model=qwen_image \
    qwen_image.drop_text_ratio=0.1 qwen_image.in_channels=32 qwen_image.patch_size=1 \
    qwen_image.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    qwen_image.eval_scheduler=Flux2Scheduler qwen_image.num_inference_steps=50 \
    qwen_image.pretrained_source=qwen_image qwen_image.pretrained_path=$path \
    qwen_image.freeze_backbone=True activation_checkpointing_mode=transformer \
    qwen_image.latent_seq_len=8192 \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=512 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    mixture.batch_size=32 mixture.resolution=512F32MS \
    mixture.data_providers=[LatentPicoBananaQwenImageGen] \
    mixture.wds_meta_dir=assets/data/meta/dc_ae_f32c32_in_1.0_512 \
    mixture.data_ext=".pth" mixture.num_workers=0 mixture.persistent_workers=False \
    eval_data_providers=[GEdit] gedit.batch_size=2 gedit.resolution=512F32MS cfg_scale=4.0 \
    lr=1e-4 clip_grad=0.1 ema_decay=null \
    warmup_steps=1000 max_steps=5000 save_checkpoint_steps=250 eval_steps=500 \
    save_eval_checkpoint_steps=1000 save_image_format=jpeg \
    save_samples_at_all_ranks=True num_save_samples=160 max_eval_steps=10 save_input_images=True \
    run_dir=exp/qwen_vae/pico_banana_qwen_image_gen/patch_head
```

### Stage 3: End-to-End Fine-Tuning on 512px

```bash
path=exp/qwen_vae/pico_banana_qwen_image_gen/patch_head/step_5000.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.trainer mode=train \
    log=False \
    resolution=512F32MS amp=bf16 model_dtype=bf16 \
    distributed_method=FSDPWrap \
    model=qwen_image \
    qwen_image.drop_text_ratio=0.1 qwen_image.in_channels=32 qwen_image.patch_size=1 \
    qwen_image.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    qwen_image.eval_scheduler=Flux2Scheduler qwen_image.num_inference_steps=50 \
    qwen_image.pretrained_source=dc-gen qwen_image.pretrained_path=$path \
    qwen_image.load_before_lora=True qwen_image.use_lora=True \
    qwen_image.lora_rank=256 qwen_image.lora_alpha=256 \
    activation_checkpointing_mode=transformer qwen_image.latent_seq_len=8192 \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=512 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    mixture.batch_size=32 mixture.resolution=512F32MS \
    mixture.data_providers=[LatentPicoBananaSingle] \
    mixture.wds_meta_dir=assets/data/meta/dc_ae_f32c32_in_1.0_512 \
    mixture.data_ext=".pth" mixture.num_workers=0 mixture.persistent_workers=False \
    eval_data_providers=[GEdit] gedit.batch_size=2 gedit.resolution=512F32MS cfg_scale=4.0 \
    lr=1e-4 clip_grad=0.1 ema_decay=null betas=[0.9,0.95] weight_decay=0.0 \
    warmup_steps=1000 max_steps=12000 save_checkpoint_steps=250 eval_steps=500 \
    save_eval_checkpoint_steps=1000 save_image_format=jpeg \
    save_samples_at_all_ranks=True num_save_samples=160 max_eval_steps=10 save_input_images=True \
    run_dir=exp/qwen_vae/pico_banana_raw/end2end
```

## Step 2: Lift Resolution from 512px to 1024px

### Stage 2.1: Merge LoRA

```bash
python -m applications.image_edit.convert_lora_into_base \
    load_path=exp/qwen_vae/pico_banana_raw/end2end/step_12000.pt \
    save_path=assets/checkpoints/image_edit/dc_gen_qwen_image_512px.pt \
    use_ema=False
```

### Stage 2.2: End-to-End Fine-Tuning on 1024px

```bash
path=assets/checkpoints/image_edit/dc_gen_qwen_image_512px.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.trainer mode=train \
    log=False \
    resolution=1024 amp=bf16 model_dtype=bf16 \
    distributed_method=FSDPWrap \
    model=qwen_image \
    qwen_image.drop_text_ratio=0.1 qwen_image.in_channels=32 qwen_image.patch_size=1 \
    qwen_image.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    qwen_image.eval_scheduler=Flux2Scheduler qwen_image.num_inference_steps=50 \
    qwen_image.pretrained_source=qwen_image qwen_image.pretrained_path=$path \
    qwen_image.load_before_lora=True qwen_image.use_lora=True \
    qwen_image.lora_rank=256 qwen_image.lora_alpha=256 \
    activation_checkpointing_mode=transformer \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=1024 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    mixture.batch_size=32 mixture.resolution=1024 \
    mixture.data_providers=[LatentPicoBananaSingle] \
    mixture.wds_meta_dir=assets/data/meta/dc_ae_f32c32_in_1.0_1024 \
    mixture.data_ext=".pth" mixture.num_workers=0 mixture.persistent_workers=False \
    eval_data_providers=[GEdit] gedit.batch_size=2 gedit.resolution=1024 cfg_scale=4.0 \
    lr=1e-4 clip_grad=0.1 ema_decay=null betas=[0.9,0.95] weight_decay=0.0 \
    warmup_steps=1000 max_steps=5000 save_checkpoint_steps=125 eval_steps=500 \
    save_eval_checkpoint_steps=1000 save_image_format=jpeg \
    save_samples_at_all_ranks=True num_save_samples=160 max_eval_steps=10 save_input_images=True \
    run_dir=exp/qwen_vae/pico_banana_raw/512to1024
```

### Stage 2.3: Merge LoRA

```bash
python -m applications.image_edit.convert_lora_into_base \
    load_path=exp/qwen_vae/pico_banana_raw/512to1024/step_1000.pt \
    save_path=assets/checkpoints/image_edit/dc_gen_qwen_image_1024px.pt \
    use_ema=False
```

## Evaluate DC-Gen Model

### Evaluate 1024px Checkpoint

```bash
path=assets/checkpoints/image_edit/dc_gen_qwen_image_1024px.pt
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.trainer mode=eval \
    resolution=1024 amp=bf16 model_dtype=bf16 \
    distributed_method=FSDPWrap \
    model=qwen_image \
    qwen_image.drop_text_ratio=0.1 qwen_image.in_channels=32 qwen_image.patch_size=1 \
    qwen_image.train_scheduler=FlowMatchEulerDiscreteSchedulerFlux2 \
    qwen_image.eval_scheduler=Flux2Scheduler qwen_image.num_inference_steps=50 \
    qwen_image.pretrained_source=qwen_image qwen_image.pretrained_path=$path \
    activation_checkpointing_mode=transformer \
    text_encoders=[qwen-image-edit/qwen2.5-vl-bf16] \
    image_encoder.resolution=1024 \
    autoencoder.name=dc-ae-f32c32-in-1.0 autoencoder.scaling_factor=0.3189 \
    eval_data_providers=[GEdit] gedit.batch_size=1 gedit.resolution=1024 cfg_scale=4.0 \
    save_samples_at_all_ranks=True num_save_samples=1212 save_input_images=True \
    run_dir=exp/qwen_vae/pico_banana/512to1024_eval
```

### Compute GEdit Score

```bash
pip install megfile qwen_vl_utils
pip install autoawq -U
```

```bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.imageeditcore.metrics_offline.compute_gedit_score \
    images_dir=exp/qwen_vae/pico_banana/512to1024_eval/0/cfg_4.0 \
    save_dir=exp/qwen_vae/pico_banana/512to1024_eval \
    resolution=1024 concat_input_output=True \
    llm=qwen25vl languages=all image_ext=.jpg
```
