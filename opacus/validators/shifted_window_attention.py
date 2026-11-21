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

from typing import List

from opacus.layers.dp_shifted_window_attention import DPShiftedWindowAttention

from .errors import ShouldReplaceModuleError, UnsupportedModuleError
from .utils import register_module_fixer, register_module_validator

try:
    import torchvision.models.swin_transformer as swin

    SWIN_AVAILABLE = True
except ImportError:
    SWIN_AVAILABLE = False


if SWIN_AVAILABLE:

    @register_module_validator(swin.ShiftedWindowAttention)
    def validate(module: swin.ShiftedWindowAttention) -> List[UnsupportedModuleError]:
        return [
            ShouldReplaceModuleError(
                "We do not support torchvision.models.swin_transformer.ShiftedWindowAttention "
                "because its implementation uses functional projections over flattened window batches "
                "and static coordinate buffers. We have written DPShiftedWindowAttention as a drop-in "
                "replacement that is fully compatible with Opacus GradSampleModule hooks and DP-SGD. "
                "Please run ModuleValidator.fix(model) to automatically replace it!"
            )
        ]

    @register_module_fixer(swin.ShiftedWindowAttention)
    def fix(module: swin.ShiftedWindowAttention) -> DPShiftedWindowAttention:
        dp_layer = DPShiftedWindowAttention(
            dim=module.qkv.in_features,
            window_size=module.window_size,
            shift_size=module.shift_size,
            num_heads=module.num_heads,
            qkv_bias=module.qkv.bias is not None,
            proj_bias=module.proj.bias is not None,
            attention_dropout=module.attention_dropout,
            dropout=module.dropout,
        )
        dp_layer.load_from(module)
        return dp_layer
