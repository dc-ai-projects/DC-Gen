# [ICLR 2025] Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models 

\[[Paper](https://arxiv.org/abs/2410.10733)\] \[[Website](https://hanlab.mit.edu/projects/dc-ae)\]

## Demo

![demo](https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/assets/dc_ae_demo.gif)
<p align="center">
<b> Figure 1: We address the reconstruction accuracy drop of high spatial-compression autoencoders.
</p>

![demo](https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/assets/dc_ae_diffusion_demo.gif)
<p align="center">
<b> Figure 2: DC-AE speeds up latent diffusion models.
</p>

<p align="left">
<img src="https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/assets/Sana-0.6B-laptop.png"  width="1200">
</p>

<p align="center">
<img src="https://huggingface.co/mit-han-lab/dc-ae-f64c128-in-1.0/resolve/main/assets/dc_ae_sana.jpg"  width="1200">
</p>

<p align="center">
<b> Figure 3: DC-AE enables efficient text-to-image generation on the laptop. For more details, please check our text-to-image diffusion model <a href="https://nvlabs.github.io/Sana/">SANA</a>.
</p>

## Abstract

We present Deep Compression Autoencoder (DC-AE), a new family of autoencoder models for accelerating high-resolution diffusion models. Existing autoencoder models have demonstrated impressive results at a moderate spatial compression ratio (e.g., 8x), but fail to maintain satisfactory reconstruction accuracy for high spatial compression ratios (e.g., 64x). We address this challenge by introducing two key techniques: (1) **Residual Autoencoding**, where we design our models to learn residuals based on the space-to-channel transformed features to alleviate the optimization difficulty of high spatial-compression autoencoders; (2) **Decoupled High-Resolution Adaptation**, an efficient decoupled three-phases training strategy for mitigating the generalization penalty of high spatial-compression autoencoders. With these designs, we improve the autoencoder's spatial compression ratio up to 128 while maintaining the reconstruction quality. Applying our DC-AE to latent diffusion models, we achieve significant speedup without accuracy drop. For example, on ImageNet 512x512, our DC-AE provides **19.1x** inference speedup and **17.9x** training speedup on H100 GPU for UViT-H while achieving a better FID, compared with the widely used SD-VAE-f8 autoencoder.
