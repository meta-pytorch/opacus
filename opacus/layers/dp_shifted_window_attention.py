#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: List[int],
) -> torch.Tensor:
    """
    Computes relative position bias from table and index.
    """
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = (
        relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    )
    return relative_position_bias


def _forward_shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    qkv_bias: Optional[Tensor],
    proj_weight: Tensor,
    proj_bias: Optional[Tensor],
    relative_position_bias_table: Tensor,
    relative_position_index: Tensor,
    window_size: List[int],
    shift_size: List[int],
    num_heads: int,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    training: bool = True,
) -> Tensor:
    """
    Pure functional forward execution of shifted window attention.
    Does not trigger PyTorch module hooks, allowing safe functional evaluation
    inside Opacus per-sample gradient samplers.
    """
    B, H, W, C = input.shape
    shift = list(shift_size)

    # Pad feature map to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    if window_size[0] >= pad_H:
        shift[0] = 0
    if window_size[1] >= pad_W:
        shift[1] = 0

    # Cyclic shift
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))

    # Partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )

    # Multi-head attention projections
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv[0], qkv[1], qkv[2]

    q = q * ((C // num_heads) ** -0.5)
    attn = q.matmul(k.transpose(-2, -1))
    rel_bias = _get_relative_position_bias(
        relative_position_bias_table, relative_position_index, window_size
    )
    attn = attn + rel_bias

    # Attention mask for shifted windows
    if sum(shift) > 0:
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift[0]),
            (-shift[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift[1]),
            (-shift[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        attn = attn.view(
            x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1)
        )
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    if attention_dropout > 0.0 and training:
        attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    if dropout > 0.0 and training:
        x = F.dropout(x, p=dropout, training=training)

    # Reverse windows
    x = x.view(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # Reverse cyclic shift
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(shift[0], shift[1]), dims=(1, 2))

    # Unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


class _LinearProxy(nn.Module):
    """
    Lightweight proxy to emulate submodules for backwards compatibility
    with external inspection code expecting `layer.qkv.weight` or `layer.proj.weight`,
    while satisfying `isinstance(proxy, nn.Module)` in `get_submodule`.
    """

    def __init__(self, weight: nn.Parameter, bias: Optional[nn.Parameter]):
        super().__init__()
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]


class DPShiftedWindowAttention(nn.Module):
    r"""
    Differentially Private Shifted Window Attention (DP-Swin) module.

    A drop-in, DP-compliant replacement for `torchvision.models.swin_transformer.ShiftedWindowAttention`.
    Standard Swin attention breaks Opacus per-sample gradient hooks because it relies on
    functional projections (`F.linear`) over window-flattened batch dimensions (`B * num_windows`),
    and contains indexed non-data buffers (`relative_position_index`).

    DPShiftedWindowAttention defines parameters directly at module scope:
    - `qkv_weight`, `qkv_bias`: Linear projection for query, key, value.
    - `proj_weight`, `proj_bias`: Linear projection for output.
    - `relative_position_bias_table`: Learnable relative positional bias.

    Coordinate buffer `relative_position_index` is managed as a constant coordinate tensor
    with device migration support in `_apply()`, avoiding unsupported buffer errors in
    Opacus `GradSampleModule.validate()`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")

        self.dim = dim
        self.window_size = list(window_size)
        self.shift_size = list(shift_size)
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.has_qkv_bias = qkv_bias
        self.has_proj_bias = proj_bias

        # Parameters declared directly on module
        self.qkv_weight = nn.Parameter(torch.empty(dim * 3, dim))
        if qkv_bias:
            self.qkv_bias = nn.Parameter(torch.empty(dim * 3))
        else:
            self.register_parameter("qkv_bias", None)

        self.proj_weight = nn.Parameter(torch.empty(dim, dim))
        if proj_bias:
            self.proj_bias = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("proj_bias", None)

        # Relative position bias table parameter
        num_rel_coords = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.empty(num_rel_coords, self.num_heads)
        )

        # Emulated linear proxies for compatibility with code expecting layer.qkv or layer.proj
        self.qkv = _LinearProxy(self.qkv_weight, self.qkv_bias)
        self.proj = _LinearProxy(self.proj_weight, self.proj_bias)

        # Coordinate index initialized as static coordinate tensor (device-tracked via _apply)
        self._init_relative_position_index()
        self.reset_parameters()

    def _init_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww

    def _apply(self, fn):
        super()._apply(fn)
        if (
            hasattr(self, "relative_position_index")
            and self.relative_position_index is not None
        ):
            self.relative_position_index = fn(self.relative_position_index)
        return self

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_weight)
        if self.qkv_bias is not None:
            nn.init.zeros_(self.qkv_bias)
        nn.init.xavier_uniform_(self.proj_weight)
        if self.proj_bias is not None:
            nn.init.zeros_(self.proj_bias)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table,
            self.relative_position_index,
            self.window_size,
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward computation matching standard ShiftedWindowAttention semantics.
        Input layout: [B, H, W, C]
        Output layout: [B, H, W, C]
        """
        return _forward_shifted_window_attention(
            input=input,
            qkv_weight=self.qkv_weight,
            qkv_bias=self.qkv_bias,
            proj_weight=self.proj_weight,
            proj_bias=self.proj_bias,
            relative_position_bias_table=self.relative_position_bias_table,
            relative_position_index=self.relative_position_index,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_heads=self.num_heads,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            training=self.training,
        )

    def load_from(self, module: nn.Module) -> "DPShiftedWindowAttention":
        """
        Loads parameters and state from a torchvision ShiftedWindowAttention module.
        """
        with torch.no_grad():
            self.qkv_weight.copy_(module.qkv.weight)
            if self.qkv_bias is not None and module.qkv.bias is not None:
                self.qkv_bias.copy_(module.qkv.bias)
            self.proj_weight.copy_(module.proj.weight)
            if self.proj_bias is not None and module.proj.bias is not None:
                self.proj_bias.copy_(module.proj.bias)
            self.relative_position_bias_table.copy_(module.relative_position_bias_table)
            if hasattr(module, "relative_position_index"):
                self.relative_position_index.copy_(module.relative_position_index)
        return self

    def state_dict(self, *args, **kwargs):
        """
        Custom state_dict supporting both native parameter keys and torchvision-style keys.
        """
        sd = super().state_dict(*args, **kwargs)
        sd["qkv.weight"] = self.qkv_weight
        if self.qkv_bias is not None:
            sd["qkv.bias"] = self.qkv_bias
        sd["proj.weight"] = self.proj_weight
        if self.proj_bias is not None:
            sd["proj.bias"] = self.proj_bias
        sd["relative_position_index"] = self.relative_position_index
        return sd

    def load_state_dict(self, state_dict, strict: bool = False):
        """
        Accepts state dicts from both torchvision and native checkpoints.
        """
        if "qkv.weight" in state_dict and "qkv_weight" not in state_dict:
            state_dict = dict(state_dict)
            state_dict["qkv_weight"] = state_dict.pop("qkv.weight")
            if "qkv.bias" in state_dict:
                state_dict["qkv_bias"] = state_dict.pop("qkv.bias")
            if "proj.weight" in state_dict:
                state_dict["proj_weight"] = state_dict.pop("proj.weight")
            if "proj.bias" in state_dict:
                state_dict["proj_bias"] = state_dict.pop("proj.bias")
        return super().load_state_dict(state_dict, strict=strict)
