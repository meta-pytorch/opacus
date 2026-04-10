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

import torch
import torch.nn as nn
from opacus.grad_sample.gsm_base import (
    AbstractGradSampleHooks,
    AbstractGradSampleModule,
)


class GradSampleHooksNoOp(AbstractGradSampleHooks):
    """
    NoOp GradSampleHooks.
    Only manages parameter attributes. The main goal of this class is to provide the same API for all methods.
    See README.md for more details
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
    ):
        if not batch_first:
            raise NotImplementedError

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )

    def remove_hooks(self) -> None:
        pass


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
    ):
        nn.Module.__init__(self)
        GradSampleHooksNoOp.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )
        self.grad_accumulation_hook = None

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return self._module.forward(x, *args, **kwargs)
