# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

## News

- \[2025/6\] 🔥 DC-AE 1.5 is accepted by ICCV 2025!

## Getting Started

```bash
conda create -n dc_gen python=3.10
conda activate dc_gen
pip install -U -r requirements.txt
```

## Content

### Deep Compression Autoencoder
- [ICLR 2025] DC-AE 1.0: [Getting Started](projects/DC-AE.md), [Website](https://hanlab.mit.edu/projects/dc-ae)
- [ICCV 2025] DC-AE 1.5: [Getting Started](projects/DC-AE-1.5.md), [Website](https://hanlab.mit.edu/projects/dc-ae-1-5), [AE Demo](https://dc-gen.hanlab.ai/), [T2I Demo](https://dc-gen.hanlab.ai/dc_gen_sana_f64c128/)

## Contact

[Han Cai](http://hancai.ai/)

## Reference

```bibtex
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
```
