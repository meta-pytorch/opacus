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
from typing import List

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleHooksFastGradientClipping,
)
from opacus.grad_sample.gsm_base import AbstractGradSampleModule
from opacus.utils.module_utils import requires_grad, trainable_parameters


logger = logging.getLogger(__name__)
logger.disabled = True


class GradSampleHooksFastGradientClippingFSDP(GradSampleHooksFastGradientClipping):
    """
    Hooks-based implementation for Fast Gradient and Ghost Clipping with FSDP support

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction="mean",
        strict: bool = True,
        max_grad_norm=1,
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
            max_grad_norm: The value at which gradients are to be clipped.
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=False,
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )

    def _get_module_type(self, module: nn.Module) -> str:
        module_type = (
            module.__class__.__bases__[1]
            if isinstance(module, torch.distributed.fsdp.FSDPModule)
            else type(module)
        )
        return module_type

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms. This is different from the parent class as norm_sample is an attribute of the module instead of the parameter."""
        norm_sample = torch.stack(
            [
                per_param_norm
                for module in self.iterate_submodules(self._module)
                for per_param_norm in module.norm_sample
            ],
            dim=0,
        ).norm(2, dim=0)

        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
        """Captures activations for the given module.
        This function is similar to the capture_activations_hook in the parent class (GradSampleModuleFastGradientClipping),
        except that it attaches _forward_counter to the module instead of parameter variable.
        Another difference is that GradSampleModuleFastGradientClipping doesn't support tied parameters only under Ghost Clipping,
        but this class doesn't supports tied parameters for either Fast Gradient Clipping or Ghost Clipping.
        """
        if (
            not requires_grad(module)
            or not module.training
            or not torch.is_grad_enabled()
            or not self.hooks_enabled
        ):
            return

        if not hasattr(module, "activations"):
            module.activations = []
        module.activations.append([t.detach() for t in forward_input])  # pyre-ignore

        if not hasattr(module, "_forward_counter"):
            module._forward_counter = 0

        module._forward_counter += 1
        if self.use_ghost_clipping and module._forward_counter > 1:
            raise NotImplementedError("Parameter tying is not supported with FSDP")

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes norms of per sample gradient given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradient norms are
        stored in ``norm_sample`` field in each module.
        This function differs from capture_backprops_hook in GradSampleModuleFastGradientClipping in that
        it attaches all the attributes to the module instead of the parameter variable.

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()

        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )

        if not hasattr(module, "norm_sample"):
            # currently, we don't support freezing and unfreezing params in between training. Making this a dictionary and mapping with param names might fix this.
            module.norm_sample = []
            for _, param in trainable_parameters(module):
                module.norm_sample.append(
                    torch.zeros(
                        torch.Size([module.max_batch_len, 1]),
                        device=param.device,
                        dtype=param.dtype,
                    )
                )

        module_type = self._get_module_type(module)
        module._forward_counter -= 1
        if self.use_ghost_clipping and module_type in self.NORM_SAMPLERS:
            norm_sampler_fn = self.NORM_SAMPLERS[module_type]
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for idx, (_, ns) in enumerate(
                (item for item in norm_samples.items() if item[0].requires_grad)
            ):
                module.norm_sample[idx] = ns
        else:
            if not self.force_functorch and module_type in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[module_type]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)

            for idx, (_, gs) in enumerate((item for item in grad_samples.items())):
                module.norm_sample[idx] = gs.reshape(len(gs), -1).norm(2, dim=-1)
            del grad_samples

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len


class GradSampleModuleFastGradientClippingFSDP(
    GradSampleHooksFastGradientClippingFSDP, AbstractGradSampleModule
):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping and FSDP support

    Computes norms of gradients without gradient instantiation
    """

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping: bool = True,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            max_grad_norm: The value at which gradients are to be clipped.
            strict: If set to True, the input module will be validated to make sure that
                it does not have buffers in all its submodules.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.
            use_ghost_clipping: If set to ``True``, Ghost Clipping
                will be used for clipping gradients of supported layers. If ``False``, Fast
                Gradient Clipping will be used for all layers.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) includes a buffer.
        """
        nn.Module.__init__(self)
        GradSampleHooksFastGradientClippingFSDP.__init__(
            self,
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            max_grad_norm=max_grad_norm,
        )
