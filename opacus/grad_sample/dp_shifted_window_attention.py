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

from typing import Dict, List

import torch
import torch.nn as nn
from torch.func import grad, vmap

from opacus.layers.dp_shifted_window_attention import (
    DPShiftedWindowAttention, _forward_shifted_window_attention)

from .utils import register_grad_sampler, register_norm_sampler


def _single_sample_swin_loss(
    single_x: torch.Tensor,
    single_grad_out: torch.Tensor,
    params_dict: Dict[str, torch.Tensor],
    relative_position_index: torch.Tensor,
    window_size: List[int],
    shift_size: List[int],
    num_heads: int,
    attention_dropout: float,
    dropout: float,
    training: bool,
) -> torch.Tensor:
    """
    Computes surrogate scalar loss for a single sample: inner product of output and incoming grad.
    Uses pure tensor function `_forward_shifted_window_attention` without triggering module hooks.
    """
    x_4d = single_x.unsqueeze(0)
    out = _forward_shifted_window_attention(
        input=x_4d,
        qkv_weight=params_dict["qkv_weight"],
        qkv_bias=params_dict.get("qkv_bias", None),
        proj_weight=params_dict["proj_weight"],
        proj_bias=params_dict.get("proj_bias", None),
        relative_position_bias_table=params_dict["relative_position_bias_table"],
        relative_position_index=relative_position_index,
        window_size=window_size,
        shift_size=shift_size,
        num_heads=num_heads,
        attention_dropout=attention_dropout,
        dropout=dropout,
        training=training,
    ).squeeze(0)
    return (out * single_grad_out).sum()


@register_grad_sampler(DPShiftedWindowAttention)
def compute_shifted_window_attention_grad_sample(
    layer: DPShiftedWindowAttention,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per-sample gradients for `DPShiftedWindowAttention`.

    Args:
        layer: The DPShiftedWindowAttention module instance.
        activations: List of activations from the forward pass. activations[0] has shape [B, H, W, C].
        backprops: Incoming gradient w.r.t layer output, with shape [B, H, W, C].

    Returns:
        Dict mapping each parameter requiring gradients to its per-sample gradient tensor with leading batch dim B.
    """
    x = activations[0]
    B = x.shape[0]

    # Collect trainable parameters dictionary
    param_map = {}
    params_dict = {}
    if layer.qkv_weight.requires_grad:
        param_map["qkv_weight"] = layer.qkv_weight
        params_dict["qkv_weight"] = layer.qkv_weight
    if layer.qkv_bias is not None and layer.qkv_bias.requires_grad:
        param_map["qkv_bias"] = layer.qkv_bias
        params_dict["qkv_bias"] = layer.qkv_bias
    if layer.proj_weight.requires_grad:
        param_map["proj_weight"] = layer.proj_weight
        params_dict["proj_weight"] = layer.proj_weight
    if layer.proj_bias is not None and layer.proj_bias.requires_grad:
        param_map["proj_bias"] = layer.proj_bias
        params_dict["proj_bias"] = layer.proj_bias
    if layer.relative_position_bias_table.requires_grad:
        param_map["relative_position_bias_table"] = layer.relative_position_bias_table
        params_dict["relative_position_bias_table"] = layer.relative_position_bias_table

    if not params_dict:
        return {}

    # Handle Poisson sampling edge-case: batch size B == 0
    if B == 0:
        ret = {}
        for p in param_map.values():
            ret[p] = torch.zeros(0, *p.shape, device=p.device, dtype=p.dtype)
        return ret

    # Define loss function parameterized by trainable weights
    def sample_loss_fn(sx, sb, p_dict):
        return _single_sample_swin_loss(
            sx,
            sb,
            p_dict,
            relative_position_index=layer.relative_position_index,
            window_size=layer.window_size,
            shift_size=layer.shift_size,
            num_heads=layer.num_heads,
            attention_dropout=layer.attention_dropout,
            dropout=layer.dropout,
            training=layer.training,
        )

    # Differentiate w.r.t parameters (argnum=2) and vectorize over batch (in_dims=(0, 0, None))
    grad_fn = grad(sample_loss_fn, argnums=2)
    batched_grad_fn = vmap(grad_fn, in_dims=(0, 0, None))

    per_sample_grads = batched_grad_fn(x, backprops, params_dict)

    ret = {}
    for name, p in param_map.items():
        ret[p] = per_sample_grads[name]

    return ret


@register_norm_sampler(DPShiftedWindowAttention)
def compute_shifted_window_attention_norm_sample(
    layer: DPShiftedWindowAttention,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per-sample gradient norms for `DPShiftedWindowAttention`.
    """
    grad_samples = compute_shifted_window_attention_grad_sample(
        layer, activations, backprops
    )
    ret = {}
    for p, g in grad_samples.items():
        B = g.shape[0]
        ret[p] = g.reshape(B, -1).norm(2, dim=-1)
    return ret
