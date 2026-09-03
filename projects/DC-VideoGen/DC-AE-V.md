## Download Models

Please download the `dc_ae_v` folder from https://drive.google.com/drive/folders/10yVeUk6MRdoF7PZPcexuiDgBaX9jJTaN to `assets/checkpoints/dc_videogen/dc_ae_v`.

## Visualize DC-AE-V Reconstruction

``` bash
python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f32t4c32-bf16 autoencoder.scaling_factor=0.7241 \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f32t4c32-bf16

python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f32t4c64-bf16 autoencoder.scaling_factor=0.7614 \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f32t4c64-bf16

python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f32t4c128-bf16 autoencoder.scaling_factor=0.4457 \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f32t4c128-bf16

python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f32t4c256-bf16 autoencoder.scaling_factor=0.4752 \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f32t4c256-bf16

python -m applications.dc_ae_v.demo_st_ae \
    autoencoder.name=dc-ae-v-1.0-f64t4c128-bf16 autoencoder.scaling_factor=0.4752 \
    input_path_list=[assets/video/kinetics_600_37_crop.mp4] h=256 w=256 t=80 \
    run_dir=exp/demo_st_ae/dc-ae-v-1.0-f64t4c128-bf16
```

## Evaluate DC-AE-V Reconstruction (TODO)

``` bash
torchrun --nnodes=1 --nproc_per_node=8 -m dc_ai.staecore.trainer \
    eval_data_providers=[ActivityNet13Test_256_256_80] base_sample_size=[256,256,16] base_batch_size=4 \
    model=dc-ae-v-1.0-f32t4c32-bf16 \
    run_dir=exp/evaluate_st_ae/dc-ae-v-1.0-f32t4c32-bf16

# expected results
# setting ActivityNet13Test_256_256_80
# latent_mean, latent_rms, latent_rms_reverse, latent_std, psnr, ssim, lpips, fvd
# 0.0086, 1.3286, 0.7527, 1.3286, 31.0825, 0.9009, 0.0447, 13.7873
```
