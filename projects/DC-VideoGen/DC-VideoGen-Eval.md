``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_t2v resolution=480 amp=bf16 model_dtype=fp32 \
    text_encoders=[wan2.1-t2v/umt5-512-bf16] wan_t2v.text_encoder_id=wan2.1-t2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_t2v.in_channels=32 wan_t2v.patch_size=[1,1,1] wan_t2v.input_size=[20,15,26] wan_t2v.depth=30 wan_t2v.hidden_size=1536 wan_t2v.num_heads=12 wan_t2v.ffn_dim=8960 \
    wan_t2v.pretrained_path=assets/checkpoints/dc_videogen/t2v/dc_videogen_wan2.1_t2v_1.3b_720p_0.1.pt wan_t2v.pretrained_source=dc-ae \
    wan_t2v.eval_scheduler=WanScheduler wan_t2v.num_inference_steps=50 wan_t2v.flow_shift=5.0 \
    run_dir=tmp
```

``` bash
python -m applications.dc_videogen.manual_prompt \
    model=wan_i2v resolution=480 amp=bf16 model_dtype=fp32 offload=True \
    image_paths=[assets/fig/wan_i2v_input.JPG] \
    text_encoders=[wan2.1-i2v/umt5-512-bf16] wan_i2v.text_encoder_id=wan2.1-i2v/umt5-512-bf16 \
    autoencoder.name=dc-ae-v-f32t4c32-1.0-bf16 autoencoder.scaling_factor=0.7241 autoencoder_dtype=bf16 \
    wan_i2v.in_channels=32 wan_i2v.patch_size=[1,1,1] wan_i2v.input_size=[20,15,26] wan_i2v.depth=40 wan_i2v.hidden_size=5120 wan_i2v.num_heads=40 wan_i2v.ffn_dim=13824 \
    wan_i2v.pretrained_path=assets/checkpoints/dc_videogen/i2v/dc_videogen_wan2.1_i2v_14b_720p_0.1.pt wan_i2v.pretrained_source=wan \
    run_dir=tmp_1
```
