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

import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import (
    AbstractGradSampleHooks,
    AbstractGradSampleModule,
)


class GradSampleHooksNoOp(AbstractGradSampleHooks):
    """
    NoOp GradSampleHooks.

    Only manages parameter attributes. Attaches to the model without wrapping it in an nn.Module.
    The main goal of this class is to provide the same API for all modes.
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
    ):
        """

        Args:
            m: nn.Module to be attached to
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to ``True``, the input module will be validated to make sure that none of its submodules includes buffers,
                which is not currently supported by Opacus.
        """
        if not batch_first:
            raise NotImplementedError

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
        )

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[NotImplementedError]:
        """
        NoOp validation.
        NoOp doesn't care about buffers or other things that usually interfere with
        per-sample gradient computation.
        """
        return []


class GradSampleModuleNoOp(GradSampleHooksNoOp, AbstractGradSampleModule):
    """
    NoOp GradSampleModule.
    Only wraps the module. The main goal of this class is to provide the same API for all methods.
    See README.md for more details
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
    ):
        nn.Module.__init__(self)
        GradSampleHooksNoOp.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
        )
        self.grad_accumulation_hook = None

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self._module.forward(x, *args, **kwargs)
