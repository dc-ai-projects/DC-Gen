# Prepare Pre-Trained Checkpoints

## Wan2.1-T2V-1.3B

```bash
hf download Wan-AI/Wan2.1-T2V-1.3B diffusion_pytorch_model.safetensors --local-dir assets/checkpoints/t2v/1.3b
python -m applications.dc_videogen.convert_checkpoint model=wan_t2v_1.3B \
    save_filename=wan_t2v_1.3b_pretrained.pt
rm assets/checkpoints/t2v/1.3b/diffusion_pytorch_model.safetensors
```

You should be able to see `assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt`.

Test: Generate videos with pre-trained checkpoints.

``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=fp32 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=wan-vae autoencoder_dtype=bf16 \
    wan_t2v.in_channels=16 wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.pretrained_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt wan_t2v.pretrained_source=wan \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 \
    run_dir=exp/demo_videogen/wan2.1_t2v_1.3b
```

## Wan2.1-14B-I2V

``` bash
hf download Wan-AI/Wan2.1-I2V-14B-480P --include "diffusion_pytorch_model-*.safetensors" --local-dir assets/checkpoints/i2v/14b
python -m applications.dc_videogen.convert_checkpoint model=wan_i2v_14B \
    file_dir=assets/checkpoints/i2v/14b \
    save_filename=wan_i2v_14b_480p_pretrained.pt \
    load_filenames=[diffusion_pytorch_model-00001-of-00007.safetensors,diffusion_pytorch_model-00002-of-00007.safetensors,diffusion_pytorch_model-00003-of-00007.safetensors,diffusion_pytorch_model-00004-of-00007.safetensors,diffusion_pytorch_model-00005-of-00007.safetensors,diffusion_pytorch_model-00006-of-00007.safetensors,diffusion_pytorch_model-00007-of-00007.safetensors]
rm assets/checkpoints/i2v/14b/diffusion_pytorch_model*.safetensors
```

You should be able to see `assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt`.

Test: Generate videos with pre-trained checkpoints.

``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_i2v resolution=480 amp=bf16 model_dtype=fp32 offload=True \
    image_paths=[assets/fig/wan_i2v_input.JPG] \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] wan_i2v.text_encoder_id=wan2.1-i2v/umt5-512-bf16 \
    autoencoder.name=wan-vae autoencoder_dtype=bf16 \
    wan_i2v.in_channels=16 wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.pretrained_path=assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt wan_i2v.pretrained_source=wan \
    run_dir=exp/demo_videogen/wan2.1_i2v_14b
```

# Prepare Training Data

Please download `fusionX_480p` from https://drive.google.com/drive/folders/1kyUG_t9r1IPr6hmb0UG9igGwy54nadph to `assets/dataset/fusionX_480p`.

``` bash
# prepare metadata for webdataset
## We only need a small subset for video embedding space alignment
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/fusionX_480p/align
## This is the complete metadata for the dataset
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/fusionX_480p

# extract video autoencoder latent feature
## We currently only provide a single tar file. To deal with more tar files, please adjust num_archives_per_task and task_id to support parallel latent feature extraction.
## Wan 2.1
python -m dc_ai.videogencore.data_provider.generate_latent_archives \
    dataset_name=Wan resolution_str=480F32MS num_frames=77 \
    meta_path=assets/dataset/fusionX_480p/align/wids-meta.json original_archive_dir=assets/dataset/fusionX_480p latent_ext=.pth \
    autoencoder.name=wan-vae \
    batch_size=1 num_archives_per_task=1 \
    save_dir=assets/dataset/latent/fusionX_480/wan_21_vae_480

## dc-ae-v-f32t4c32-1.0-bf16
python -m dc_ai.videogencore.data_provider.generate_latent_archives \
    dataset_name=Wan resolution_str=480F32MS num_frames=80 \
    meta_path=assets/dataset/fusionX_480p/wids-meta.json original_archive_dir=assets/dataset/fusionX_480p latent_ext=.pth \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 dtype=bf16 \
    batch_size=1 num_archives_per_task=1 \
    save_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480

# prepare metadata again for the extracted latent archives
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/wan_21_vae_480/align save_path=assets/data/meta/wids/wan_21_vae_480/fusionX_align.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480 save_path=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480/fusionX.json
python -m dc_ai.apps.data_provider.web_dataset.generate_meta data_dir=assets/dataset/latent/fusionX_480/dc_ae_v_f32t4c32_1.0_480/align save_path=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480/fusionX_align.json
```

Check training data

``` bash
python -m dc_ai.dc_adapt.data_provider.latent_mixture \
    data_providers=[LatentFusionXAlign] data_ext=.pth \
    teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    teacher_autoencoder.name=wan-vae \
    student_autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 student_autoencoder.scaling_factor=0.7241 \
    batch_size=2 resolution=480F32MS \
    save_checkpoint_steps=10 distributed_method=FSDP
```

You should be able to see paired video reconstructions in `tmp_mixture`, like `tmp_mixture/sample_0_0_dc_ae_v_f32t4c32.mp4` and `tmp_mixture/sample_0_0_wan_vae.mp4`.

# Train DC-VideoGen-Wan2.1-T2V-1.3B

## Phase 1: Video Embedding Space Alignment

``` bash
python -m dc_ai.dc_adapt.trainer mode=train \
    dataset=latent_video data_type=latent model_type=wan_t2v \
    latent_video.data_providers=[LatentFusionXAlign] latent_video.data_ext=.pth \
    latent_video.teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    latent_video.student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    latent_video.batch_size=4 \
    image_encoder.resolution=480 \
    teacher_stae.name=wan-vae \
    student_stae.name=dc-ae-v-f32t4c32-1.0-bf16 student_stae.scaling_factor=0.7241 \
    teacher_wan_t2v.input_size=[20,60,104] teacher_wan_t2v.patch_size=[1,2,2] \
    teacher_wan_t2v.in_channels=16 teacher_wan_t2v.hidden_size=1536 \
    teacher_wan_t2v.pretrained_source=wan teacher_wan_t2v.pretrained_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt \
    student_wan_t2v.input_size=[20,15,26] student_wan_t2v.patch_size=[1,1,1] \
    student_wan_t2v.in_channels=32 student_wan_t2v.hidden_size=1536 \
    align_type=downsample upsample_mode=nearest \
    lr_scheduler=WSDCosineLR warmup_steps=0 stable_steps=60000 \
    optimizer=AdamW lr=2e-4 weight_decay=0 lr_scheduler=WSDCosineLR ema_decay=null betas=[0.9,0.9] amp=bf16 distributed_method=DDP \
    save_checkpoint_steps=1000 save_samples_steps=null eval_steps=5000 max_steps=20000 \
    run_dir=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_1 \
    log=False
```

## Phase 2: Patch-Head Fine-Tuning

``` bash
# assemble aligned batch with pre-trained backbone
path=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_1
checkpoint=step_20000.pt
python -m applications.dc_videogen.wan_t2v_assemble \
    patch_size=[1,1,1] input_size=[20,15,26] in_channels=32 \
    depth=30 hidden_size=1536 num_heads=12 \
    backbone_checkpoint_path=assets/checkpoints/t2v/1.3b/wan_t2v_1.3b_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/phase_2_init.pt

# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_samples=32 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False vbench_text_prompt.batch_size=1 \
    category_ids=[0] save_samples_at_all_ranks=True \
    mixture.resolution=480F32MS mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=fp32 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=$path/phase_2_init.pt wan_t2v.freeze_backbone=True \
    optimizer=AdamW lr=0.0001 betas=[0.9,0.999] lr_scheduler=ConstantLRwithWarmup clip_grad=0.1 weight_decay=0.0 warmup_steps=0 warmup_lr=0.0 \
    amp=bf16 activation_checkpointing_mode=transformer \
    max_steps=6000 eval_steps=2000 \
    distributed_method=FSDPWrap \
    run_dir=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_2 log=False
```

## Phase 3: End-to-End Fine-Tuning

``` bash
checkpoint=step_6000.pt
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchTextPrompt] \
    vbench_text_prompt.category_id=12 vbench_text_prompt.num_samples=32 vbench_text_prompt.num_workers=0 vbench_text_prompt.persistent_workers=False vbench_text_prompt.batch_size=1 \
    category_ids=[0,1,2,3,4,5,6,7,8,9,10] save_samples_at_all_ranks=True \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    model=wan_t2v model_dtype=fp32 \
    wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] \
    wan_t2v.pretrained_source=dc-ae wan_t2v.pretrained_path=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_2/$checkpoint \
    wan_t2v.freeze_text_embed=True wan_t2v.freeze_cross_attn=True \
    wan_t2v.train_scheduler=FlowMatchScheduler wan_t2v.flow_shift=8.0 \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 cfg_scale=3.0 \
    use_lora=True lora_rank=256 lora_alpha=512 \
    optimizer=AdamW lr=0.00005 betas=[0.9,0.999] lr_scheduler=ConstantLRwithWarmup warmup_lr=0.0 clip_grad=0.1 weight_decay=0.0 warmup_steps=1000 ema_decay=0.9999 \
    distributed_method=FSDPWrap \
    amp=bf16 activation_checkpointing_mode=transformer \
    max_steps=100000 eval_steps=2000 \
    run_dir=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3 log=False

# merge LoRA weights
python -m applications.dc_videogen.convert_lora_into_base \
    load_path=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_100.pt \
    save_path=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_100_merged.pt
```

## Evaluation

``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=fp32 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.pretrained_path=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_100_merged.pt wan_t2v.pretrained_source=dc-ae \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 \
    run_dir=exp/dc_videogen/fusionX/wan2.1_t2v_1.3B_to_dc_ae_v_f32t4c32_1.0/phase_3
```

# Train DC-VideoGen-Wan2.1-I2V-14B

## Phase 1: Video Embedding Space Alignment

``` bash
python -m dc_ai.dc_adapt.trainer mode=train \
    dataset=latent_video data_type=latent model_type=wan_i2v \
    latent_video.data_providers=[LatentFusionXAlign] latent_video.data_ext=.pth \
    latent_video.teacher_wds_meta_dir=assets/data/meta/wids/wan_21_vae_480 \
    latent_video.student_wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 \
    latent_video.batch_size=4 \
    teacher_stae.name=wan-vae \
    student_stae.name=dc-ae-v-f32t4c32-1.0-bf16 student_stae.scaling_factor=0.7241 \
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
    run_dir=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1 \
    log=False
```

## Phase 2: Patch-Head Fine-Tuning

``` bash
# assemble aligned batch with pre-trained backbone
path=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_1
checkpoint=step_20000.pt
python -m applications.dc_videogen.wan_i2v_assemble \
    patch_size=[1,1,1] input_size=[20,15,26] in_channels=32 \
    depth=40 hidden_size=5120 num_heads=40 ffn_dim=13824 \
    backbone_checkpoint_path=assets/checkpoints/i2v/14b/wan_i2v_14b_480p_pretrained.pt \
    patch_embedding_checkpoint_path=$path/$checkpoint \
    checkpoint_export_path=$path/phase_2_init.pt

# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchImagePrompt] \
    vbench_image_prompt.num_samples=32 skip_vbench_evaluator=True category_ids=[0] \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    save_samples_at_all_ranks=True \
    model=wan_i2v model_dtype=fp32 \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_source=dc-ae wan_i2v.pretrained_path=$path/phase_2_init.pt wan_i2v.freeze_backbone=True \
    wan_i2v.train_scheduler=FlowMatchScheduler wan_i2v.flow_shift=5.0 \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 \
    optimizer=AdamW lr=0.0001 weight_decay=0.0 betas=[0.9,0.999] clip_grad=0.1 warmup_steps=0 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=6000 eval_steps=200000 save_eval_checkpoint_steps=2000 save_checkpoint_steps=200 \
    distributed_method=FSDPWrap \
    run_dir=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2 log=False
```

## Phase 3: End-to-End Fine-Tuning

``` bash
checkpoint=step_6000.pt
# here is an example for one-node training, in practice we run this experiments on two nodes
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.videogencore.trainer mode=train \
    resolution=480F32 eval_data_providers=[VBenchImagePrompt] \
    mixture.resolution=480F32MS \
    mixture.data_providers=[LatentFusionX] mixture.wds_meta_dir=assets/data/meta/wids/dc_ae_v_f32t4c32_1.0_480 mixture.num_workers=0 mixture.persistent_workers=False mixture.batch_size=2 \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    save_samples_at_all_ranks=True \
    model=wan_i2v model_dtype=fp32 \
    wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] \
    wan_i2v.pretrained_source=dc-ae wan_i2v.pretrained_path=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_2/$checkpoint \
    wan_i2v.freeze_text_embed=True wan_i2v.freeze_cross_attn=True \
    wan_i2v.train_scheduler=FlowMatchScheduler wan_i2v.flow_shift=5.0 \
    wan_i2v.eval_scheduler=WanScheduler wan_i2v.num_inference_steps=50 cfg_scale=5.0 \
    use_lora=True lora_rank=256 lora_alpha=512 \
    optimizer=AdamW lr=0.00005 betas=[0.9,0.999] clip_grad=0.1 weight_decay=0.0 warmup_steps=1000 ema_decay=0.9999 warmup_lr=0.0 lr_scheduler=ConstantLRwithWarmup \
    distributed_method=FSDPWrap \
    amp=bf16 activation_checkpointing_mode=transformer offload=True \
    max_steps=100000 eval_steps=200000 save_eval_checkpoint_steps=1000 save_checkpoint_steps=100 \
    run_dir=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3 log=False

# merge LoRA weights
python -m applications.dc_videogen.convert_lora_into_base \
    load_path=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_10.pt \
    save_path=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_10_merged.pt
```

## Evaluation

``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_i2v resolution=480 amp=bf16 model_dtype=fp32 offload=True \
    image_paths=[assets/fig/wan_i2v_input.JPG] \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] wan_i2v.text_encoder_id=wan2.1-i2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.pretrained_path=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3/step_10_merged.pt wan_i2v.pretrained_source=dc-ae \
    run_dir=exp/dc_videogen/fusionX/wan2.1_i2v_14B_to_dc_ae_v_f32t4c32_1.0/phase_3
```
