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

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleHooksFastGradientClipping,
)
from opacus.grad_sample.gsm_base import AbstractGradSampleModule


logger = logging.getLogger(__name__)
logger.disabled = False


class GradSampleHooksFastGradientClippingTP(GradSampleHooksFastGradientClipping):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: torch.nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping=True,
    ):
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=use_ghost_clipping,
        )
        self.set_pattern_param_sample_norm_sum()
        warnings.warn(
            "Opacus TP is currently in beta. Custom model output placements may cause unexpected behavior."
        )

    def set_pattern_param_sample_norm_sum(self):
        """
        When initializing the gradient sample module, the following function investigates the tensor parallelism state of different model parameters, and then decide whether or not to merge the per-sample gradient norm from local devices.

        Specifically, under the following exceptions, we should not merge the per-sample norm from local devices, but maintain the one from device 0:
        1. The parameter is not a DTensor.
        2. The model is ``nn.embedding`` while under ``RowWiseParallel``.
        3. The model weight is not sharded. For example, ``nn.linear.bias`` under ``RowWiseParallel``. This situation is currently unsupported since ``fsdpoptimizer`` requires all the parameters to be sharded.
        """
        for module in self.iterate_submodules(self._module):
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if type(param) is not torch.distributed.tensor.DTensor:
                        param.merge_flag = False
                    elif type(module) is torch.nn.Embedding and param.placements[
                        0
                    ].is_shard(0):
                        param.merge_flag = False
                    elif param.placements[0].is_replicate():
                        param.merge_flag = False
                        raise NotImplementedError(
                            "We currently do not support replicated model parameters, due to the constraint of fsdpoptimizwer. This means that nn.Linear.bias must be disabled or configured to be sharded."
                        )
                    else:
                        param.merge_flag = True

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        current_rank = torch.distributed.get_rank()

        squared_norm_sample = (
            torch.stack(
                [
                    (
                        torch.zeros_like(param._norm_sample)
                        if (not param.merge_flag and current_rank != 0)
                        else param._norm_sample
                    )
                    for param in self.trainable_parameters
                ],
                dim=0,
            )
            .norm(2, dim=0)
            .square()
        )

        torch.distributed.all_reduce(
            squared_norm_sample, op=torch.distributed.ReduceOp.SUM
        )
        self.per_sample_gradient_norms = squared_norm_sample.sqrt()
        return squared_norm_sample.sqrt()


class GradSampleModuleFastGradientClippingTP(
    GradSampleHooksFastGradientClippingTP, AbstractGradSampleModule
):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: torch.nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping=True,
    ):
        nn.Module.__init__(self)
        GradSampleHooksFastGradientClippingTP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=use_ghost_clipping,
        )
        self.grad_accumulation_hook = None
