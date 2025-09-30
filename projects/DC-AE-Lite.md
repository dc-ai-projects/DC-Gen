# DC-AE-Lite
Decoding is often the speed bottleneck in few-step latent diffusion models. We release [dc-ai format](https://huggingface.co/dc-ai/dc-ae-lite-f32c32), [diffusers format](https://huggingface.co/dc-ai/dc-ae-lite-f32c32-diffusers). It has the same encoder of DC-AE-f32c32-SANA-1.0 while having a much smaller decoder, achieving **1.8x faster** decoding. Without training, it can be applied to diffusion model trained with DC-AE-f32c32-SANA-1.0.

## Demo
<p align="center">
  <img src="https://huggingface.co/strangerTHU/dc-ae-lite-figures/resolve/main/combined.gif"><br>
  <b> DC-AE-Lite vs DC-AE reconstruction visual quality </b>
</p>

<p align="center">
  <img src="https://huggingface.co/strangerTHU/dc-ae-lite-figures/resolve/main/dc-ae-lite.jpg"><br>
  <b> DC-AE-Lite achieves 1.8× faster decoding than DC-AE with similar reconstruction quality </b>
</p>