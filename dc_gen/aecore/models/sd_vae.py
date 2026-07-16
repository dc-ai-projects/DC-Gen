# SD-VAE was introduced by Robin Rombach*, Andreas Blattmann*, Dominik Lorenz, Patrick Esser, and Björn Ommer in "High-Resolution Image Synthesis with Latent Diffusion Models", see https://arxiv.org/abs/2112.10752.
# The original implementation is by Machine Vision and Learning Group, LMU Munich, licensed under the MIT License. See https://github.com/CompVis/latent-diffusion.

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from omegaconf import MISSING, OmegaConf
from torch import nn
from torch.nn import functional as F

from ...models.utils.network import get_submodule_weights

from .base import BaseAE, BaseAEConfig


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    num_groups = min(num_groups, in_channels)
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    MODE: str = "flash"

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        if AttnBlock.MODE == "flash":
            B, C, H, W = q.shape
            q, k, v = q.reshape(B, C, H * W), k.reshape(B, C, H * W), v.reshape(B, C, H * W)
            q, k, v = q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1)
            out = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous())
            out = out.permute(0, 2, 1)
            h_ = out.reshape(B, C, H, W)
        else:
            # compute attention
            b, c, h, w = q.shape
            q = q.reshape(b, c, h * w)
            q = q.permute(0, 2, 1)  # b,hw,c
            k = k.reshape(b, c, h * w)  # b,c,hw
            w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w_ = w_ * (int(c) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)

            # attend to values
            v = v.reshape(b, c, h * w)
            w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
            h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla"], f"attn_type {attn_type} unknown"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)


@dataclass
class SDVAEEncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int] = ()
    dropout: float = 0.0
    resamp_with_conv: bool = True
    resolution: int = 256
    double_z: bool = True
    attn_type: str = "vanilla"


class SDVAEEncoder(nn.Module):
    def __init__(self, cfg: SDVAEEncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.temb_ch = 0
        self.num_resolutions = len(cfg.ch_mult)
        self.num_res_blocks = cfg.num_res_blocks

        # downsampling
        self.conv_in = torch.nn.Conv2d(cfg.in_channels, cfg.ch, kernel_size=3, stride=1, padding=1)

        curr_res = cfg.resolution
        in_ch_mult = (1,) + tuple(cfg.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = cfg.ch * in_ch_mult[i_level]
            block_out = cfg.ch * cfg.ch_mult[i_level]
            for _ in range(cfg.num_res_blocks):
                if i_level == 3:
                    print(f"block_in {block_in}, block_out {block_out}")
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=cfg.dropout
                    )
                )
                block_in = block_out
                if curr_res in cfg.attn_resolutions:
                    print(f"ldm_encoder: make attention at level {i_level}")
                    attn.append(make_attn(block_in, attn_type=cfg.attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, cfg.resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=cfg.attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.dropout
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * cfg.latent_channels if cfg.double_z else cfg.latent_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


@dataclass
class SDVAEDecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING
    ch: int = 128
    out_ch: int = 3
    ch_mult: tuple[int] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int] = ()
    dropout: float = 0.0
    resamp_with_conv: bool = True
    resolution: int = 256
    give_pre_end: bool = False
    tanh_out: bool = False
    attn_type: str = "vanilla"


class SDVAEDecoder(nn.Module):
    def __init__(self, cfg: SDVAEDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.temb_ch = 0
        self.num_resolutions = len(cfg.ch_mult)
        self.num_res_blocks = cfg.num_res_blocks

        block_in = cfg.ch * cfg.ch_mult[self.num_resolutions - 1]
        curr_res = cfg.resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, cfg.latent_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(cfg.latent_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.dropout
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=cfg.attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=cfg.dropout
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = cfg.ch * cfg.ch_mult[i_level]
            for _ in range(cfg.num_res_blocks + 1):
                if i_level == 3:
                    print(f"block_in {block_in}, block_out {block_out}")
                block.append(
                    ResnetBlock(
                        in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=cfg.dropout
                    )
                )
                block_in = block_out
                if curr_res in cfg.attn_resolutions:
                    print(f"ldm_decoder: make attention at level {i_level}")
                    attn.append(make_attn(block_in, attn_type=cfg.attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, cfg.resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, cfg.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.cfg.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.cfg.tanh_out:
            h = torch.tanh(h)
        return h


@dataclass
class SDVAEConfig(BaseAEConfig):
    in_channels: int = 3
    latent_channels: int = 4

    encoder: SDVAEEncoderConfig = field(
        default_factory=lambda: SDVAEEncoderConfig(
            in_channels="${..in_channels}", latent_channels="${..latent_channels}"
        )
    )
    decoder: SDVAEDecoderConfig = field(
        default_factory=lambda: SDVAEDecoderConfig(
            in_channels="${..in_channels}", latent_channels="${..latent_channels}"
        )
    )

    sample_posterior: bool = False
    use_quant_conv: bool = True

    attn_mode: str = "flash"

    pretrained_path: Optional[str] = None
    pretrained_source: str = "ldm"
    load_pretrained_ema: bool = True


class SDVAE(BaseAE):
    def __init__(self, cfg: SDVAEConfig):
        super().__init__(cfg)
        self.cfg: SDVAEConfig

        self.encoder = SDVAEEncoder(cfg.encoder)
        self.decoder = SDVAEDecoder(cfg.decoder)

        if self.cfg.use_quant_conv:
            if self.encoder.cfg.double_z:
                self.quant_conv = nn.Conv2d(2 * cfg.latent_channels, 2 * cfg.latent_channels, 1)
            else:
                self.quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)
            self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, 1)
        AttnBlock.MODE = cfg.attn_mode

        if self.cfg.pretrained_path is not None:
            self.load_model()

    @property
    def spatial_compression_ratio(self) -> int:
        return 2 ** (self.decoder.num_resolutions - 1)

    def load_model(self):
        if self.cfg.pretrained_source in ["ldm", "mar"]:
            if self.cfg.pretrained_source == "ldm":
                state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["state_dict"]
            elif self.cfg.pretrained_source == "mar":
                state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)["model"]
            self.encoder.load_state_dict(get_submodule_weights(state_dict, "encoder."))
            self.decoder.load_state_dict(get_submodule_weights(state_dict, "decoder."))
            self.quant_conv.load_state_dict(get_submodule_weights(state_dict, "quant_conv."))
            self.post_quant_conv.load_state_dict(get_submodule_weights(state_dict, "post_quant_conv."))
        elif self.cfg.pretrained_source == "dc-ae":
            state_dict = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
            if "ema" in state_dict and self.cfg.load_pretrained_ema:
                state_dict = next(iter(state_dict["ema"].values()))
            else:
                state_dict = state_dict["model_state_dict"]
            if self.cfg.gan_weight == 0:
                for key in list(state_dict.keys()):
                    if key.startswith("1."):
                        state_dict.pop(key)
            self.get_trainable_modules_list().load_state_dict(state_dict)
        else:
            raise ValueError(f"pretrained_source {self.cfg.pretrained_source} is not supported")

    def encode(
        self, input: torch.Tensor, latent_channels: Optional[list[int]] = None
    ) -> torch.Tensor | list[torch.Tensor]:
        latent = self.encoder(input)

        if self.cfg.use_quant_conv:
            latent = self.quant_conv(latent)
        if self.encoder.cfg.double_z:
            posterior = DiagonalGaussianDistribution(latent)
            if self.cfg.sample_posterior:
                latent = posterior.sample()
            else:
                latent = posterior.mode()
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if self.cfg.use_quant_conv:
            latent = self.post_quant_conv(latent)
        output = self.decoder(latent)
        return output


def sd_vae_f8(name: str, pretrained_path: str) -> SDVAE:
    if name == "sd-vae-f8":
        cfg_str = "latent_channels=4 " "encoder.ch_mult=[1,2,4,4] " "decoder.ch_mult=[1,2,4,4]"
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: SDVAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SDVAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    model = SDVAE(cfg)
    return model


def sd_vae_f16(name: str, pretrained_path: str) -> SDVAE:
    if name == "sd-vae-f16":
        cfg_str = (
            "latent_channels=16 "
            "encoder.ch_mult=[1,1,2,2,4] encoder.attn_resolutions=[16] "
            "decoder.ch_mult=[1,1,2,2,4] decoder.attn_resolutions=[16]"
        )
    elif name == "mar-vae-f16":
        cfg_str = (
            "latent_channels=16 "
            "encoder.ch_mult=[1,1,2,2,4] encoder.attn_resolutions=[16] "
            "decoder.ch_mult=[1,1,2,2,4] decoder.attn_resolutions=[] "
            "attn_mode=vanilla "
            "pretrained_source=mar"
        )
    elif name == "vavae-imagenet256-f16d32-dinov2":
        cfg_str = (
            "latent_channels=32 "
            "encoder.ch_mult=[1,1,2,2,4] encoder.attn_resolutions=[16] "
            "decoder.ch_mult=[1,1,2,2,4] decoder.attn_resolutions=[16]"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: SDVAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SDVAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    model = SDVAE(cfg)
    return model


def sd_vae_f32(name: str, pretrained_path: str) -> SDVAE:
    if name == "sd-vae-f32":
        cfg_str = (
            "latent_channels=64 "
            "encoder.ch_mult=[1,1,2,2,4,4] encoder.attn_resolutions=[16,8] "
            "decoder.ch_mult=[1,1,2,2,4,4] decoder.attn_resolutions=[16,8]"
        )
    else:
        raise NotImplementedError
    cfg = OmegaConf.from_dotlist(cfg_str.split(" "))
    cfg: SDVAEConfig = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(SDVAEConfig), cfg))
    cfg.pretrained_path = pretrained_path
    model = SDVAE(cfg)
    return model
