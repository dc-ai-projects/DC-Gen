# [ICCV 2025] DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space

\[[Paper]()\] \[[AE Demo](https://dc-ae-1-5.hanlab.ai/)\] \[[Website](https://hanlab.mit.edu/projects/dc-ae-1-5)\]

## News

- \[2025/6\] 🔥 DC-AE 1.5 is accepted by ICCV 2025!

## Abstract

We present DC-AE 1.5, a new family of deep compression autoencoders for high-resolution diffusion models. Increasing the autoencoder's latent channel number is a highly effective approach for improving its reconstruction quality. However, it results in slow convergence for diffusion models, leading to poorer generation quality despite better reconstruction quality. This issue limits the quality upper bound of latent diffusion models and hinders the employment of autoencoders with higher spatial compression ratios. We introduce two key innovations to address this challenge: i) **Structured Latent Space**, a training-based approach to impose a desired channel-wise structure on the latent space with front latent channels capturing object structures and latter latent channels capturing image details; ii) **Augmented Diffusion Training**, an augmented diffusion training strategy with additional diffusion training objectives on object latent channels to accelerate convergence. With these techniques, DC-AE 1.5 delivers faster convergence and better diffusion scaling results than DC-AE. On ImageNet 512x512, DC-AE-1.5-f64c128 delivers better image generation quality than DC-AE-f32c32 while being **4x** faster.

## Release
- The code and pretrained models will be released after the legal review is completed.
