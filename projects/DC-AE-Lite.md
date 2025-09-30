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

## Usage
#### Diffusers Implementation

```bash
pip install -U diffusers
```

```python
from diffusers import AutoencoderDC
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

device = torch.device("cuda")
dc_ae_lite = AutoencoderDC.from_pretrained("dc-ai/dc-ae-lite-f32c32-diffusers").to(device).eval()

transform = transforms.Compose([
    transforms.CenterCrop((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image = Image.open("assets/fig/girl.png")

x = transform(image)[None].to(device)
latent = dc_ae_lite.encode(x).latent
print(f"latent shape: {latent.shape}")

y = dc_ae_lite.decode(latent).sample
save_image(y * 0.5 + 0.5, "demo_dc_ae_lite.png")
```
#### Our Implementation
```bash
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dc_gen.ae_model_zoo import DCAE_HF

device = torch.device("cuda")
dc_ae_lite = DCAE_HF.from_pretrained("dc-ai/dc-ae-lite-f32c32").to(device).eval()

transform = transforms.Compose([
    transforms.CenterCrop((1024,1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

image = Image.open("assets/fig/girl.png")

x = transform(image)[None].to(device)
latent = dc_ae_lite.encode(x)
print(f"latent shape: {latent.shape}")

y = dc_ae_lite.decode(latent)
save_image(y * 0.5 + 0.5, "demo_dc_ae_lite.png")
```