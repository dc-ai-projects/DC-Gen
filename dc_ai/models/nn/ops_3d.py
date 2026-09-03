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

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import get_same_padding, get_submodule_weights, val2tuple
from .act import build_act
from .norm import (
    TritonRMSNorm2d,
    build_norm,
)
from .ops import IdentityLayer, OpSequential, ResidualBlock


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
        self.spatial_padding_mode = spatial_padding_mode
        self.temporal_padding_mode = temporal_padding_mode
        if causal:
            self.custom_padding = (0, 0, 0, 0, 2 * padding[0], 0)
            padding = (0, padding[1], padding[2])
            self.custom_padding_mode = "constant" if temporal_padding_mode == "zeros" else temporal_padding_mode
        elif causal_chunk_length is not None:
            assert spatial_padding_mode == "zeros"
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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str = "zero_pad"):
        if method == "zero_pad":
            nn.init.constant_(self.conv.weight, 0)
            if self.causal:
                self.conv.weight.data[:, :, -1] = state_dict["conv.weight"]
            else:
                self.conv.weight.data[:, :, self.conv.weight.data.shape[2] // 2] = state_dict["conv.weight"]
        elif method == "split":
            self.conv.weight.data.copy_(state_dict["conv.weight"][:, :, None] / self.conv.weight.shape[2])
        else:
            raise ValueError(f"init method {method} is not supported")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            self.conv.bias.data = state_dict["conv.bias"]
        if self.norm:
            self.norm.load_state_dict(get_submodule_weights(state_dict, "norm."))
        if self.act:
            self.act.load_state_dict(get_submodule_weights(state_dict, "act."))

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if self.custom_padding is not None:
            x = F.pad(x, self.custom_padding, mode=self.custom_padding_mode)

        if self.causal_chunk_length is not None:
            #  TODO: implement video-image joint training
            B, C, T, H, W = x.shape
            assert (
                T % self.causal_chunk_length == 0
            ), f"T={T} is not divisible by self.causal_chunk_length={self.causal_chunk_length}"
            assert self.conv.stride[0] == 1
            x = x.reshape(B, C, T // self.causal_chunk_length, self.causal_chunk_length, H, W).transpose(
                1, 2
            )  # (B, T // self.causal_chunk_length, C, self.causal_chunk_length, H, W)

            if feature_cache is not None:
                first_left_pad = feature_cache[feature_key] if feature_key in feature_cache else None
                feature_cache[feature_key] = x[:, -1:, :, -self.conv.padding[0] :].clone().detach()
            else:
                first_left_pad = None
            if first_left_pad is None:
                if self.temporal_padding_mode == "zeros":
                    first_left_pad = torch.zeros((B, 1, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device)
                elif self.temporal_padding_mode == "replicate":
                    first_left_pad = x[:, :1, :, :1, :, :].repeat(
                        (1, 1, 1, self.conv.padding[0], 1, 1)
                    )  # (B, 1, C, self.conv.padding[0], H, W)
                else:
                    raise ValueError(f"temporal padding mode {self.temporal_padding_mode} is not supported")
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
                    assert self.temporal_padding_mode == "zeros"
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
            if self.temporal_padding_mode == "zeros":
                right_pad = torch.zeros(
                    (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W), dtype=x.dtype, device=x.device
                )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            elif self.temporal_padding_mode == "replicate":
                right_pad = x[:, :, :, -1:].repeat(
                    (1, 1, 1, self.conv.padding[0], 1, 1)
                )  # (B, T // self.causal_chunk_length, C, self.conv.padding[0], H, W)
            else:
                raise ValueError(f"temporal padding mode {self.temporal_padding_mode} is not supported")
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
        elif self.causal:
            if feature_cache is not None:
                if feature_key in feature_cache:
                    x[:, :, : self.custom_padding[4]] = feature_cache[feature_key]
                feature_cache[feature_key] = x[:, :, -self.custom_padding[4] :].clone().detach()
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x, feature_cache

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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv1.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv1."), method)
        self.conv2.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv2."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x, feature_cache_1 = self.conv1(x, feature_cache, feature_key + "conv1." if feature_key is not None else None)
        x, feature_cache_2 = self.conv2(x, feature_cache, feature_key + "conv2." if feature_key is not None else None)
        if feature_cache is not None:
            assert feature_cache_1 is not None and feature_cache_2 is not None
            returned_feature_cache = feature_cache_1 | feature_cache_2
        else:
            returned_feature_cache = None
        return x, returned_feature_cache


def pixel_unshuffle_3d(x: torch.Tensor, spatial_factor: int, temporal_factor: int) -> torch.Tensor:
    # x: (B, C, T, H, W)
    B, C, T, H, W = x.shape
    assert (
        T % temporal_factor == 0 and W % spatial_factor == 0 and H % spatial_factor == 0
    ), f"{T=}, {W=}, {H=}, {spatial_factor=}, {temporal_factor=} are not supported"
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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x, returned_feature_cache = self.conv(
            x, feature_cache, feature_key + "conv." if feature_key is not None else None
        )
        x = pixel_unshuffle_3d(x, self.spatial_factor, self.temporal_factor)
        if feature_cache is not None:
            assert returned_feature_cache is not None
        return x, returned_feature_cache


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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        pass

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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.conv.load_state_dict_from_2d(get_submodule_weights(state_dict, "conv."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        x, returned_feature_cache = self.conv(
            x, feature_cache, feature_key + "conv." if feature_key is not None else None
        )
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        if feature_cache is not None:
            assert returned_feature_cache is not None
        return x, returned_feature_cache


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

    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = pixel_shuffle_3d(x, self.spatial_factor, self.temporal_factor)
        return x


class ResidualBlock3d(ResidualBlock):
    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        self.main.load_state_dict_from_2d(get_submodule_weights(state_dict, f"main."), method)
        if isinstance(self.shortcut, (IdentityLayer,)):
            pass
        else:
            self.shortcut.load_state_dict_from_2d(get_submodule_weights(state_dict, f"shortcut."), method)

    def forward_main(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
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
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if self.main is None:
            res = x
            returned_feature_cache = None if feature_cache is None else {}
        elif self.shortcut is None:
            res, returned_feature_cache = self.forward_main(x, feature_cache, feature_key)
        else:
            res_main, returned_feature_cache = self.forward_main(x, feature_cache, feature_key)
            res_shortcut = self.shortcut(x)
            res = res_main + res_shortcut
            if self.post_act:
                res = self.post_act(res)
        if feature_cache is not None:
            assert returned_feature_cache is not None
        return res, returned_feature_cache


class OpSequential3d(OpSequential):
    def load_state_dict_from_2d(self, state_dict: dict[str, torch.Tensor], method: str):
        for i, op in enumerate(self.op_list):
            if isinstance(op, (TritonRMSNorm2d, nn.SiLU)):
                op.load_state_dict(get_submodule_weights(state_dict, f"op_list.{i}."))
            else:
                op.load_state_dict_from_2d(get_submodule_weights(state_dict, f"op_list.{i}."), method)

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: Optional[dict[str, torch.Tensor]] = None,
        feature_key: Optional[str] = None,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        returned_feature_cache = None if feature_cache is None else {}
        for i, op in enumerate(self.op_list):
            if isinstance(op, torch.distributed.algorithms._checkpoint.checkpoint_wrapper.CheckpointWrapper):
                op_class = type(op._checkpoint_wrapped_module)
            else:
                op_class = type(op)
            if issubclass(
                op_class,
                (
                    ConvLayer3d,
                    ResidualBlock3d,
                    ConvPixelUnshuffleDownSampleLayer3d,
                    ConvPixelShuffleUpSampleLayer3d,
                ),
            ):
                x, feature_cache_i = op(
                    x, feature_cache, feature_key + f"op_list.{i}." if feature_key is not None else None
                )
                if feature_cache is not None:
                    assert feature_cache_i is not None
                    returned_feature_cache |= feature_cache_i
            elif issubclass(
                op_class,
                (
                    TritonRMSNorm2d,
                    nn.SiLU,
                    IdentityLayer,
                ),
            ):
                x = op(x)
            else:
                raise ValueError(f"Unsupported op class: {op_class}")
        return x, returned_feature_cache
