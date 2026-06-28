# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import collections
from functools import partial
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_same_padding(kernel_size: int | tuple[int, ...]) -> int | tuple[int, ...]:
    if isinstance(kernel_size, (tuple, list)):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def get_submodule_weights(weights: collections.OrderedDict, prefix: str):
    submodule_weights = collections.OrderedDict()
    len_prefix = len(prefix)
    for key, weight in weights.items():
        if key.startswith(prefix):
            submodule_weights[key[len_prefix:]] = weight
    return submodule_weights


def val2list(x: list | tuple | Any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list | tuple | Any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


REGISTERED_ACT_DICT: dict[str, type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
    "mish": nn.Mish,
    "linear": nn.Identity,
    "lrelu": partial(nn.LeakyReLU, negative_slope=0.2),
}


def build_act(name: Optional[str]) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        return act_cls()
    else:
        return None


class RMSNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = x / torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        x = x.permute(0, 3, 1, 2)
        return x


class RMSNorm3d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = x / torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        x = x.permute(0, 4, 1, 2, 3)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": nn.BatchNorm2d,
    "ln": nn.LayerNorm,
    "rms2d": RMSNorm2d,
    "rms3d": RMSNorm3d,
}


def build_norm(name: Optional[str] = "bn2d", num_features=None, **kwargs) -> Optional[nn.Module]:
    if name in ["ln", "rms2d", "rms3d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        return norm_cls(**kwargs)
    else:
        return None


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class OpSequential(nn.Module):
    def __init__(self, op_list: list[Optional[nn.Module]]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


def conv3d_split_channel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: int | Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    num_in_channel_chunks: int,
    num_out_channel_chunks: int,
) -> torch.Tensor:
    out_channels, in_channels = weight.shape[0], weight.shape[1]
    assert in_channels % num_in_channel_chunks == 0 and out_channels % num_out_channel_chunks == 0
    in_channels_per_split = in_channels // num_in_channel_chunks
    out_channels_per_split = out_channels // num_out_channel_chunks

    output = []
    for i in range(num_out_channel_chunks):
        out_channels_start, out_channels_end = i * out_channels_per_split, (i + 1) * out_channels_per_split
        output_i = 0
        for j in range(num_in_channel_chunks):
            in_channels_start, in_channels_end = j * in_channels_per_split, (j + 1) * in_channels_per_split
            x_j = x[:, in_channels_start:in_channels_end]
            weight_j = weight[out_channels_start:out_channels_end, in_channels_start:in_channels_end]
            output_i = output_i + F.conv3d(x_j, weight_j, stride=stride, padding=padding, dilation=dilation, groups=1)
        output.append(output_i)
    output = torch.cat(output, dim=1)
    if bias is not None:
        output = output + bias[:, None, None, None]
    return output


def custom_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: Sequence[int],
    padding: int | Sequence[int],
    dilation: int | Sequence[int],
    groups: int,
) -> torch.Tensor:
    input_sample_numel = input[0].numel()
    output_sample_numel = (
        weight.shape[0] * (input.shape[2] // stride[0]) * (input.shape[3] // stride[1]) * (input.shape[4] // stride[2])
    )

    if (input_sample_numel >= 1 << 31 or output_sample_numel >= 1 << 31) and groups == 1:
        num_in_channel_chunks, num_out_channel_chunks = 1, 1
        while input_sample_numel // num_in_channel_chunks >= 1 << 31:
            num_in_channel_chunks *= 2
        while output_sample_numel // num_out_channel_chunks >= 1 << 31:
            num_out_channel_chunks *= 2
        # print(f"num_in_channel_chunks {num_in_channel_chunks}, num_out_channel_chunks {num_out_channel_chunks}")
        output = conv3d_split_channel(
            input, weight, bias, stride, padding, dilation, num_in_channel_chunks, num_out_channel_chunks
        )
        return output
    else:
        return F.conv3d(input, weight, bias, stride, padding, dilation, groups)


class CustomConv3d(nn.Conv3d):
    def _conv_forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        padding: Optional[tuple[int, ...]] = None,
    ):
        assert self.padding_mode == "zeros"
        return custom_conv3d(input, weight, bias, self.stride, padding or self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor, padding: Optional[tuple[int, ...]] = None) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias, padding)


class ConvLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...] = 3,
        stride: int | tuple[int, ...] = 1,
        groups: int = 1,
        use_bias: bool = False,
        norm: Optional[str] = "bn2d",
        act_func: Optional[str] = "relu",
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = val2tuple(kernel_size, 3)
        stride = val2tuple(stride, 3)
        padding = get_same_padding(kernel_size)
        self.causal = causal
        self.causal_chunk_length = causal_chunk_length
        if causal:
            self.custom_padding = (0, 0, 0, 0, 2 * padding[0], 0)
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = "constant" if temporal_padding_mode == "zeros" else temporal_padding_mode
        elif causal_chunk_length is not None:
            assert spatial_padding_mode == temporal_padding_mode == "zeros"
            self.custom_padding = None
            self.custom_padding_mode = None
        elif spatial_padding_mode != temporal_padding_mode:
            self.custom_padding = (0, 0, 0, 0, padding[0], padding[0])
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = "constant" if temporal_padding_mode == "zeros" else temporal_padding_mode
        else:
            self.custom_padding = None
            self.custom_padding_mode = None
        self.conv = CustomConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=use_bias,
            padding_mode=spatial_padding_mode,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

        self.zero_out = zero_out
        if zero_out:
            if self.norm:
                self.norm.zero_out()
            else:
                nn.init.constant_(self.conv.weight, 0)
                nn.init.constant_(self.conv.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        if self.custom_padding is not None:
            x = F.pad(x, self.custom_padding, mode=self.custom_padding_mode)

        if self.causal_chunk_length is not None and x.shape[2] % self.causal_chunk_length == 0:
            B, C, T, H, W = x.shape
            assert T % self.causal_chunk_length == 0
            assert self.conv.stride[0] == 1
            x = x.reshape(B, C, T // self.causal_chunk_length, self.causal_chunk_length, H, W).transpose(
                1, 2
            )  # (B, T // self.causal_chunk_length, C, self.causal_chunk_length, H, W)

            if feature_cache is not None:
                first_left_pad = feature_cache[feature_key] if feature_key in feature_cache else None
                feature_cache[feature_key] = x[:, -1:, :, -self.conv.padding[0] :].clone()
            else:
                first_left_pad = None
            if first_left_pad is None:
                first_left_pad = torch.zeros((B, 1, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device)
            else:
                assert (
                    first_left_pad.shape[0] == B
                    and first_left_pad.shape[1] == 1
                    and first_left_pad.shape[2] == C
                    and first_left_pad.shape[3] <= self.conv.padding[0]
                    and first_left_pad.shape[4] == H
                    and first_left_pad.shape[5] == W
                )
                if first_left_pad.shape[3] < self.conv.padding[0]:
                    first_left_pad = torch.cat(
                        [
                            torch.zeros(
                                (B, 1, C, self.conv.padding[0] - first_left_pad.shape[3], H, W),
                                dtype=x.dtype,
                                device=x.device,
                            ),
                            first_left_pad,
                        ],
                        dim=3,
                    )  # (B, 1, C, self.conv.padding[0], H, W)

            left_pad = torch.cat(
                [first_left_pad, x[:, :-1, :, -self.conv.padding[0] :]], dim=1
            )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            right_pad = torch.zeros(
                (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device
            )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            x = torch.cat(
                [left_pad, x, right_pad], dim=3
            )  # (B, T // self.causal_chunk_length, C, self.causal_chunk_length + 2 * self.conv.padding[0], H, W)
            x = x.reshape(
                B * (T // self.causal_chunk_length), C, self.causal_chunk_length + 2 * self.conv.padding[0], H, W
            )
            x = self.conv(
                x, (0, self.conv.padding[1], self.conv.padding[2])
            )  # (B * (T // self.causal_chunk_length), C, self.causal_chunk_length, H, W)
            x = (
                x.reshape(B, T // self.causal_chunk_length, -1, self.causal_chunk_length, H, W)
                .transpose(1, 2)
                .reshape(B, -1, T, H, W)
            )  # (B, C, T // self.causal_chunk_length, self.causal_chunk_length, H, W)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

    def __repr__(self):
        _str = f"{self.__class__.__name__}(\n" f"  (conv): {self.conv}\n"
        if self.norm:
            _str += f"  (norm): {self.norm}\n"
        if self.act:
            _str += f"  (act): {self.act}\n"
        _str += f"  zero_out={self.zero_out}\n"
        _str += f"  causal={self.causal}\n"
        _str += f"  causal_chunk_length={self.causal_chunk_length}\n"
        _str += f")"
        return _str


class DSConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        use_bias: bool | tuple[bool, bool] = False,
        norm: Optional[str] | tuple[Optional[str], Optional[str]] = "trms2d",
        act_func: Optional[str] | tuple[Optional[str], Optional[str]] = ("silu", None),
        zero_out: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        kernel_size = val2tuple(kernel_size, 3)
        stride = val2tuple(stride, 3)

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer3d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            causal_chunk_length=causal_chunk_length,
        )
        self.point_conv = ConvLayer3d(
            in_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            zero_out=zero_out,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.depth_conv(x, feature_cache, feature_key + "depth_conv." if feature_key is not None else None)
        x = self.point_conv(x, feature_cache, feature_key + "point_conv." if feature_key is not None else None)
        return x


class ResBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: bool | tuple[bool, bool] = False,
        norm: str | tuple[Optional[str], Optional[str]] = ("bn2d", "bn2d"),
        act_func: str | tuple[Optional[str], Optional[str]] = ("relu6", None),
        zero_out: bool = False,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = round(in_channels * expand_ratio) if mid_channels is None else mid_channels

        self.conv1 = ConvLayer3d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )
        self.conv2 = ConvLayer3d(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
            zero_out=zero_out,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv1(x, feature_cache, feature_key + "conv1." if feature_key is not None else None)
        x = self.conv2(x, feature_cache, feature_key + "conv2." if feature_key is not None else None)
        return x


def pixel_unshuffle_3d(x: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    assert T % temporal_factor == 0 and W % spatial_factor == 0 and H % spatial_factor == 0
    x = (
        x.reshape(
            (
                B,
                C,
                T // temporal_factor,
                temporal_factor,
                H // spatial_factor,
                spatial_factor,
                W // spatial_factor,
                spatial_factor,
            )
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(
            B, C * temporal_factor * spatial_factor**2, T // temporal_factor, H // spatial_factor, W // spatial_factor
        )
    )
    return x


def pixel_shuffle_3d(x: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    assert C % (temporal_factor * spatial_factor**2) == 0
    x = (
        x.reshape(
            (B, C // temporal_factor // spatial_factor**2, temporal_factor, spatial_factor, spatial_factor, T, H, W)
        )
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(
            B, C // temporal_factor // spatial_factor**2, T * temporal_factor, H * spatial_factor, W * spatial_factor
        )
    )
    return x


class ConvPixelUnshuffleDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        assert out_channels % out_ratio == 0
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels // out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv(x, feature_cache, feature_key + "conv." if feature_key is not None else None)
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        assert in_channels * spatial_factor**2 * temporal_factor % out_channels == 0
        self.group_size = in_channels * spatial_factor**2 * temporal_factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        B, C, T, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, T, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        spatial_factor: int,
        temporal_factor: int,
        spatial_padding_mode: str = "zeros",
        temporal_padding_mode: str = "zeros",
        zero_out: bool = False,
        causal: bool = False,
        causal_chunk_length: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        out_ratio = spatial_factor**2 * temporal_factor
        self.conv = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels * out_ratio,
            kernel_size=kernel_size,
            use_bias=True,
            norm=None,
            act_func=None,
            spatial_padding_mode=spatial_padding_mode,
            temporal_padding_mode=temporal_padding_mode,
            zero_out=zero_out,
            causal=causal,
            causal_chunk_length=causal_chunk_length,
        )

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        x = self.conv(x, feature_cache, feature_key + "conv." if feature_key is not None else None)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class ChannelDuplicatingPixelShuffleUpSampleLayer3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_factor: int,
        temporal_factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_factor = spatial_factor
        self.temporal_factor = temporal_factor
        assert out_channels * spatial_factor**2 * temporal_factor % in_channels == 0
        self.repeats = out_channels * spatial_factor**2 * temporal_factor // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class ResidualBlock3d(ResidualBlock):
    def forward_main(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        feature_key = feature_key + "main." if feature_key is not None else None
        if self.pre_norm is None:
            return self.main(x, feature_cache, feature_key)
        else:
            return self.main(self.pre_norm(x), feature_cache, feature_key)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x, feature_cache, feature_key)
        else:
            res = self.forward_main(x, feature_cache, feature_key) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


class OpSequential3d(OpSequential):
    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> torch.Tensor:
        for i, op in enumerate(self.op_list):
            if isinstance(op, (ConvLayer3d, ResidualBlock3d, ConvPixelShuffleUpSampleLayer3d)):
                x = op(x, feature_cache, feature_key + f"op_list.{i}." if feature_key is not None else None)
            else:
                x = op(x)
        return x
