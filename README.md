# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

<div align="center">
  <a href="https://hanlab.mit.edu/projects/dc-gen/"><img src="https://img.shields.io/static/v1?label=Website&message=DC-Gen&color=darkred&logo=github-pages"></a> &ensp;
  <a href="https://arxiv.org/abs/2509.25180"><img src="https://img.shields.io/static/v1?label=arXiv&message=DC-Gen&color=red&logo=arxiv"></a> &ensp;
  <a href="https://kxn6bflf-dc-gen-demo.xenon.lepton.run/"><img src="https://img.shields.io/static/v1?label=Demo&message=DC-Gen&color=blue&logo=googlechrome"></a> &ensp;
  <!-- <a href="https://huggingface.co/collections/dc-ai/dc-gen-6899bb095082244f396203e1"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=DC-AI&color=yellow&logo=huggingface"></a> &ensp; -->
</div>

## 📖 Overview

<p align="center" border-radius="10px">
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/teaser2.png" width="100%" alt="teaser"/>
</p>

<p align="center">
  <em>DC-Gen delivers high-quality, high-resolution visual generation across text-to-image, text-to-video, image-to-video, and image editing tasks, achieving up to 53.8× speedup over pre-trained diffusion models.</em>
</p>

## 🎬 Overview Video

[![Watch the video](https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/demo_video_first_frame.jpg)](https://www.youtube.com/watch?v=gu9KKtJgbho)

## 🔥🔥 News

- (🔥 New) \[2026/6/18\] DC-Gen is accepted by ECCV 2026! 🎉
- (🔥 New) \[2025/9/30\] We released the [DC-Gen technical report](https://arxiv.org/abs/2509.25180) on arXiv.
- (🔥 New) \[2025/9/30\] We released [DC-AE-Lite](projects/DC-AE-Lite.md).
- \[2025/6\] DC-AE 1.5 is accepted by ICCV 2025!

## 💡 Introduction

DC-Gen is a new acceleration framework for diffusion models. DC-Gen works with any pre-trained diffusion model, boosting efficiency by transferring it into a deeply compressed latent space with lightweight post-training. For example, applying DC-Gen to FLUX.1-Krea takes just 31 H100 GPU days. The resulting DC-Gen-FLUX delivers the same quality as the base model while achieving dramatic gains—53.8× faster inference on H100 at 4K resolution. DC-Gen has been validated across text-to-image, text-to-video, image-to-video, and image editing tasks.

### Highlight 1: DC-Gen Maintains Base Model Quality with Faster Inference Speed

Modern visual diffusion models are recognized for their superior realism and text-rendering capabilities but suffer from low throughput. DC-Gen models successfully preserve these qualities while delivering a significant speedup over corresponding base models.

- <ins>Comparison to Previous Models on 1024×1024 Resolution.</ins>
<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/appendix_1K.jpg" alt="teaser_page4"/>
</figure>

- <ins>Comparison to pre-trained Wan2.1 on 720×1280 Resolution, 81 frames.</ins>

**Text-to-Video**

<table>
<tr>
<td align="center"><small>Wan2.1-T2V-14B</small><br>(<font color="red"><b>27.52</b></font> mins/video)</td>
<td align="center"><small>DC-Gen-Wan2.1-T2V-14B</small><br>(<font color="red"><b>3.58</b></font> mins/video)</td>
<td align="center"><small>Wan2.1-T2V-14B</small><br>(<font color="red"><b>27.52</b></font> mins/video)</td>
<td align="center"><small>DC-Gen-Wan2.1-T2V-14B</small><br>(<font color="red"><b>3.58</b></font> mins/video)</td>
</tr>
<tr>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_wan_3.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_dcgen_3.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_wan_4.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_dcgen_4.gif" width="100%"/></td>
</tr>
<tr>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_wan_2.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_dcgen_2.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_wan_5.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/t2v_dcgen_5.gif" width="100%"/></td>
</tr>
</table>

**Image-to-Video**

<table>
<tr>
<td align="center"><small>Wan2.1-I2V-14B</small><br>(<font color="red"><b>27.88</b></font> mins/video)</td>
<td align="center"><small>DC-Gen-Wan2.1-I2V-14B</small><br>(<font color="red"><b>3.67</b></font> mins/video)</td>
<td align="center"><small>Wan2.1-I2V-14B</small><br>(<font color="red"><b>27.88</b></font> mins/video)</td>
<td align="center"><small>DC-Gen-Wan2.1-I2V-14B</small><br>(<font color="red"><b>3.67</b></font> mins/video)</td>
</tr>
<tr>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_wan_1.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_dcgen_1.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_wan_2.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_dcgen_2.gif" width="100%"/></td>
</tr>
<tr>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_wan_3.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_dcgen_3.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_wan_4.gif" width="100%"/></td>
<td><img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comparisons/i2v_dcgen_4.gif" width="100%"/></td>
</tr>
</table>

- <ins>Comparison to pre-trained Qwen-Image-Edit on 1K image editing tasks.</ins>
<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/appendix_qwen_1k.jpg" width="100%" alt="qwen_edit_comparison"/>
</figure>

- <ins>The relative speedup of DC-Gen is more significant at higher resolutions, achieving up to 53.8× acceleration on DC-Gen-FLUX.</ins>
<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comp_speed.png" alt="teaser_page3"/>
</figure>

### Highlight 2: DC-Gen Enables Native Super-Resolution Image Generation with Exceptional Efficiency

- <ins>FLUX and Z-Image do not support native 4K image generation due to prohibitive training and inference costs. DC-Gen-FLUX and DC-Gen-Z-Image address this limitation by reducing token redundancy with [DC-AE-f64c128](projects/DC-AE.md).</ins>
<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/comp_qualitative.jpg" alt="teaser_page2"/>
</figure>

### Highlight 3: DC-Gen Facilitates Rapid and Stable Autoencoder Adaptation

Previously, changing the autoencoder required retraining diffusion models from scratch, which was highly inefficient. DC-Gen introduces **Embedding Alignment** to transfer the base model's knowledge to the new latent space. After this alignment, the model can perform visual generation with correct semantics in the new latent space without fine-tuning the diffusion model's weights. With only a few steps of end-to-end fine-tuning, the DC-Gen models can already achieve visual generation quality comparable to pre-trained models.

<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/analysis.png" alt="teaser_page5"/>
</figure>

Following embedding alignment, we can fully recover the quality through LoRA fine-tuning.
<figure>
  <img src="https://huggingface.co/dc-ai/DC-Gen-README/resolve/main/assets/pipeline.png" alt="teaser_page5"/>
</figure>

## Getting Started

```bash
conda create -n dc_ai python=3.10
conda activate dc_ai
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 # please install torch according to your cuda version
# for arm architecture, use
# pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu129
pip install -U -r requirements.txt
pip install flash-attn --no-build-isolation
# if you failed to import flash_attn, please uninstall and build from source
# pip uninstall flash_attn
# pip install ninja
# git clone git@github.com:Dao-AILab/flash-attention.git
# cd flash-attention
# python setup.py install

pre-commit install
```

## Offline Inference

Apart from the Online Demo we provided, you can also run DC-Gen locally with a simple command-line interface. This codebase supports text-to-image (1K and 4K), text-to-video, image-to-video, and instruction-based image editing. See [applications/dc_gen/inference_demo/README.md](applications/dc_gen/inference_demo/README.md) for details.

## Accelerate Pre-trained Diffusion Models with DC-Gen

We provide step-by-step guides for applying DC-Gen to FLUX, Qwen-Image-Edit, Wan2.1, and Z-Image, covering data preparation, embedding alignment, and LoRA fine-tuning. See [projects/DC-Gen](projects/DC-Gen) for details.

## Contact

[Han Cai](http://hancai.ai/)

## Related Projects

- [ICLR 2025] DC-AE 1.0: [Getting Started](projects/DC-AE.md), [Website](https://hanlab.mit.edu/projects/dc-ae)
- [ICCV 2025] DC-AE 1.5: [Getting Started](projects/DC-AE-1.5.md), [Website](https://hanlab.mit.edu/projects/dc-ae-1-5), [AE Demo](https://dc-gen.hanlab.ai/), [T2I Demo](https://dc-gen.hanlab.ai/dc_gen_sana_f64c128/)
- DC-AE-Lite: [Getting Started](projects/DC-AE-Lite.md)

## Reference

```bibtex
@article{he2025dc,
  title={DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space},
  author={He, Wenkun and Gu, Yuchao and Chen, Junyu and Zou, Dongyun and Lin, Yujun and Zhang, Zhekai and Xi, Haocheng and Li, Muyang and Zhu, Ligeng and Yu, Jincheng and others},
  journal={arXiv preprint arXiv:2509.25180},
  year={2025}
}

@article{chen2024deep,
  title={Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models},
  author={Chen, Junyu and Cai, Han and Chen, Junsong and Xie, Enze and Yang, Shang and Tang, Haotian and Li, Muyang and Lu, Yao and Han, Song},
  journal={arXiv preprint arXiv:2410.10733},
  year={2024}
}

@article{chen2025dc,
  title={DC-AE 1.5: Accelerating Diffusion Model Convergence with Structured Latent Space},
  author={Chen, Junyu and Zou, Dongyun and He, Wenkun and Chen, Junsong and Xie, Enze and Han, Song and Cai, Han},
  journal={arXiv preprint arXiv:2508.00413},
  year={2025}
}

@misc{zou2025dcaelite,
  title  = {DC-AE-Lite},
  author = {Zou, Dongyun and Chen, Junyu and He, Wenkun and Chen, Junsong and Xie, Enze and Han, Song and Cai, Han},
  url    = {https://github.com/dc-ai-projects/DC-Gen/blob/main/projects/DC-AE-Lite.md},
  month  = Sep,
  year   = {2025}
}
```
