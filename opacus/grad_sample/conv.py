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

import math
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.tensor_utils import unfold2d, unfold3d

from .utils import register_grad_sampler


@register_grad_sampler([nn.Conv1d, nn.Conv2d, nn.Conv3d])
def compute_conv_grad_sample(
    layer: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]

    activations = activations.to(backprops.dtype)

    n = activations.shape[0]
    if n == 0:
        # Empty batch
        ret = {}
        ret[layer.weight] = torch.zeros_like(layer.weight).unsqueeze(0)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.zeros_like(layer.bias).unsqueeze(0)
        return ret

    # FSDPWrapper adds a prefix 'FSDP' to layer type, e.g. FSDPConv2d.
    # Therefore the layer type can not be directly determined by type(layer).
    layer_type = (
        layer.__class__.__bases__[1]
        if isinstance(layer, torch.distributed.fsdp.FSDPModule)
        else type(layer)
    )
    padding = layer.padding
    if layer.padding_mode != "zeros":
        # Non-zero padding modes are applied to the input before the convolution
        # (see ``torch.nn.modules.conv._ConvNd._conv_forward``), so do the same
        # here and unfold the padded input without any extra zero padding.
        activations = F.pad(
            activations,
            layer._reversed_padding_repeated_twice,
            mode=layer.padding_mode,
        )
        padding = (0,) * len(layer.kernel_size)
    # get activations and backprops in shape depending on the Conv layer
    if layer_type is nn.Conv2d:
        activations = unfold2d(
            activations,
            kernel_size=layer.kernel_size,
            padding=padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    elif layer_type is nn.Conv1d:
        activations = activations.unsqueeze(-2)  # add the H dimension
        # set arguments to tuples with appropriate second element
        if padding == "same":
            total_pad = layer.dilation[0] * (layer.kernel_size[0] - 1)
            left_pad = math.floor(total_pad / 2)
            right_pad = total_pad - left_pad
        elif padding == "valid":
            left_pad, right_pad = 0, 0
        else:
            left_pad, right_pad = padding[0], padding[0]
        activations = F.pad(activations, (left_pad, right_pad))
        activations = torch.nn.functional.unfold(
            activations,
            kernel_size=(1, layer.kernel_size[0]),
            stride=(1, layer.stride[0]),
            dilation=(1, layer.dilation[0]),
        )
    elif layer_type is nn.Conv3d:
        activations = unfold3d(
            activations,
            kernel_size=layer.kernel_size,
            padding=padding,
            stride=layer.stride,
            dilation=layer.dilation,
        )
    backprops = backprops.reshape(n, -1, activations.shape[-1])

    ret = {}
    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret


# @register_grad_sampler([nn.Conv2d])
def convolution2d_backward_as_a_convolution(
    layer: nn.Conv2d,
    activations: List[torch.Tensor],
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for Conv2d layers using backward.
    This is an alternative implementation and is not used because it is slower in many contexts.

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0]
    batch_size = activations.shape[0]
    input_size = activations.shape[1]
    output_size = backprops.shape[1]

    # activations has shape (B, I, H, W)
    # backprops has shape (B, O, H, W)
    activations_ = activations.view(
        batch_size,
        layer.groups,
        input_size // layer.groups,
        activations.shape[2],
        activations.shape[3],
    )  # (B, G, I/G, H, W)

    activations_ = activations_.view(
        activations_.shape[0] * activations_.shape[1],
        activations_.shape[2],
        activations_.shape[3],
        activations_.shape[4],
    )  # (B*G, I / G, H, W)
    activations_ = activations_.transpose(0, 1)  # (I / G, B * G, H, W)
    backprops_ = backprops.view(
        backprops.shape[0] * backprops.shape[1],
        1,
        backprops.shape[2],
        backprops.shape[3],
    )  # (B*O, 1, H, W)

    # Without groups (I, B, H, W) X (B*O, 1, H, W) -> (I, B*O, H, W)
    # With groups (I / G, B*G, H, W) X (B*O, 1, H, W) -> (I / G, B * O, H, W)
    weight_grad_sample = F.conv2d(
        activations_,
        backprops_,
        bias=None,
        dilation=layer.stride,
        padding=layer.padding,
        stride=layer.dilation,
        groups=batch_size * layer.groups,
    )
    weight_grad_sample = weight_grad_sample.view(
        input_size // layer.groups,
        batch_size,
        output_size,
        *weight_grad_sample.shape[-2:]
    )  # (I / G, B, O, H, W)
    weight_grad_sample = weight_grad_sample.movedim(0, 2)  # (B, O, I/G, H, W)
    weight_grad_sample = weight_grad_sample[
        :, :, :, : layer.weight.shape[2], : layer.weight.shape[3]
    ]

    ret = {layer.weight: weight_grad_sample}
    if layer.bias is not None:
        ret[layer.bias] = torch.sum(backprops, dim=[-1, -2])

    return ret
