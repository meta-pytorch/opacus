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
from opacus.grad_sample.grad_sample_module import (
    GradSampleModule,
    create_or_accumulate_grad_sample,
    promote_current_grad_sample,
)
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN
from opacus.utils.module_utils import (
    requires_grad,
    trainable_modules,
    trainable_parameters,
)


logger = logging.getLogger(__name__)
logger.disabled = True


def create_norm_sample(
    *, param: torch.Tensor, grad_sample: torch.Tensor, max_batch_len: int
) -> None:
    """
    Creates a ``_norm_sample`` attribute in the given parameter


    Args:
        param: Parameter to which ``_norm_sample`` will be added
        grad_sample: Per-sample gradients tensor. Must be of the same
            shape as ``param`` with extra batch dimension
    """

    if param.requires_grad:
        if (
            max_batch_len == 0
        ):  # To handle the case of empty batch that may arise from Poisson sampling
            param._norm_sample = torch.tensor(
                [], device=grad_sample.device, dtype=grad_sample.dtype
            )
        else:
            param._norm_sample = torch.zeros(
                torch.Size([max_batch_len, 1]),
                device=grad_sample.device,
                dtype=grad_sample.dtype,
            )
            param._norm_sample = grad_sample.reshape(len(grad_sample), -1).norm(
                2, dim=-1
            )


class GradSampleModuleFastGradientClipping(GradSampleModule):
    """
    Hooks-based implementation of GradSampleModule with Fast Gradient and Ghost Clipping

    Computes norms of gradients without gradient instantiation
    """

    NORM_SAMPLERS = {}

    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        max_grad_norm=1,
        use_ghost_clipping=True,
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
        if logger.isEnabledFor(logging.INFO):
            self.log_module_gradient_sample_mode(
                module=m,
                force_functorch=force_functorch,
                use_ghost_clipping=use_ghost_clipping,
            )

        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch,
        )
        self.trainable_parameters = [p for _, p in trainable_parameters(self._module)]
        self.max_grad_norm = max_grad_norm
        self.use_ghost_clipping = use_ghost_clipping
        self._per_sample_gradient_norms = None

    def get_clipping_coef(self) -> torch.Tensor:
        """Get per-example gradient scaling factor for clipping."""
        norm_sample = self.get_norm_sample()
        return (self.max_grad_norm / (norm_sample + 1e-6)).clamp(max=1.0)

    def get_norm_sample(self) -> torch.Tensor:
        """Get per-example gradient norms."""
        norm_samples = [param._norm_sample for param in self.trainable_parameters]
        if norm_samples:
            target_device = norm_samples[0].device
            norm_samples = [norm.to(target_device) for norm in norm_samples]
        norm_sample = torch.stack(norm_samples, dim=0).norm(2, dim=0)
        self.per_sample_gradient_norms = norm_sample
        return norm_sample

    def capture_activations_hook(
        self,
        module: nn.Module,
        forward_input: List[torch.Tensor],
        _forward_output: torch.Tensor,
    ):
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

        for _, p in trainable_parameters(module):
            p._forward_counter += 1
            if (
                self.use_ghost_clipping
                and p._forward_counter > 1
                and type(module) in self.NORM_SAMPLERS
            ):
                raise NotImplementedError(
                    "Parameter tying is not supported with Ghost Clipping"
                )

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
        stored in ``norm_sample`` field in each parameter.

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
        activations = [
            temp.to_local() if type(temp) is torch.distributed.tensor.DTensor else temp
            for temp in activations
        ]

        if self.use_ghost_clipping and type(module) in self.NORM_SAMPLERS:
            norm_sampler_fn = self.NORM_SAMPLERS[type(module)]
            norm_samples = norm_sampler_fn(module, activations, backprops)

            for param, ns in norm_samples.items():
                if param.requires_grad:
                    param._norm_sample = ns
                    param._forward_counter -= 1

        else:
            if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
                grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
            else:
                grad_sampler_fn = ft_compute_per_sample_gradient

            grad_samples = grad_sampler_fn(module, activations, backprops)
            for param, gs in grad_samples.items():
                create_or_accumulate_grad_sample(
                    param=param, grad_sample=gs, max_batch_len=module.max_batch_len
                )
            del grad_samples
            # Detect end of current batch processing and switch accumulation
            # mode from sum to stacking. Used for RNNs and tied parameters
            # (See #417 for details)
            for _, p in trainable_parameters(module):
                p._forward_counter -= 1
                if p._forward_counter == 0:
                    promote_current_grad_sample(p)
                    create_norm_sample(
                        param=p,
                        grad_sample=p.grad_sample,
                        max_batch_len=module.max_batch_len,
                    )
                    p.grad_sample = None
        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def log_module_gradient_sample_mode(
        self, module: nn.Module, *, force_functorch=False, use_ghost_clipping=True
    ):
        """
        Add logs to track gradient sample mode for each part of the module, including 1) Ghost Clipping, 2) Fast Gradient Clipping (hook mode), and 3) Fast Gradient Clipping (functorch mode).

        Args:
            module: nn.Module to be checked
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.
            use_ghost_clipping: If set to ``True``, Ghost Clipping
                will be used for clipping gradients of supported layers. If ``False``, Fast
                Gradient Clipping will be used for all layers.
        """
        for m_name, m in trainable_modules(module):
            if type(m) in [DPRNN, DPLSTM, DPGRU]:
                logger.info(
                    f"Module name: {m_name}, module type: {type(m)}. No hook or functorch is added."
                )

            elif use_ghost_clipping and type(m) in self.NORM_SAMPLERS:
                logger.info(
                    f"Module name: {m_name}, module type: {type(m)}, under Ghost Clipping."
                )

            else:
                if not force_functorch and type(m) in self.GRAD_SAMPLERS:
                    # When functorch is not enforced, use FGC (hook mode) if the layer has a registered grad_sampler (supported). Otherwise, use FGC (functorch mode).
                    logger.info(
                        f"Module name: {m_name}, module type: {type(m)}, under Fast Gradient Clipping (hook mode)."
                    )
                else:
                    logger.info(
                        f"Module name: {m_name}, module type: {type(m)}, under Fast Gradient Clipping (functorch mode)."
                    )

    @property
    def per_sample_gradient_norms(self) -> torch.Tensor:
        """Returns per sample gradient norms. Note that these are not privatized and should only be used for debugging purposes or in non-private settings"""
        if self._per_sample_gradient_norms is not None:
            return self._per_sample_gradient_norms
        else:
            raise AttributeError(
                "per_sample_gradient_norms is not set. Please call forward and backward on the model before accessing this property."
            )

    @per_sample_gradient_norms.setter
    def per_sample_gradient_norms(self, value):
        self._per_sample_gradient_norms = value
