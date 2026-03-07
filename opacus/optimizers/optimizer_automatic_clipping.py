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

from typing import List

import torch
from opacus.optimizers.optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _mark_as_processed,
)
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer


class DPAutomaticClippingOptimizer(DPOptimizer):
    """
    DPOptimizer variant that uses automatic clipping across all layers.

    Automatic clipping computes per-sample clip factors using the formula:
        ``per_sample_clip_factor = max_grad_norm / (per_sample_norms + 0.01)``

    This differs from the default behavior by using automatic scaling
    (without clamping to 1.0) as described in:
    *"Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger"*
    https://arxiv.org/pdf/2206.07136

    The stabilization constant (0.01) prevents division by zero for samples with
    very small gradients, ensuring numerical stability during training.

    Note:
        This optimizer is automatically instantiated when using
        ``PrivacyEngine.make_private()`` with ``clipping="automatic"``.

    See Also:
        - :class:`~opacus.optimizers.optimizer.DPOptimizer`: Base DP optimizer with standard clipping
        - :class:`~DPPerLayerAutomaticClippingOptimizer`: Per-layer variant of automatic clipping
        - :class:`~opacus.optimizers.ddpoptimizer_automatic_clipping.DistributedDPAutomaticClippingOptimizer`: Distributed version
    """

    def clip_and_accumulate(self):
        """Perform automatic clipping and accumulate clipped gradients.

        This method mirrors :meth:`DPOptimizer.clip_and_accumulate` but uses the
        automatic clipping formula for per-sample clip factors.
        """

        # Compute per-parameter norms (shape: [batch_size] for each parameter)
        per_param_norms: List[torch.Tensor] = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
        ]

        if per_param_norms:
            target_device = per_param_norms[0].device
            per_param_norms = [norm.to(target_device) for norm in per_param_norms]

            # Combine per-parameter norms to per-sample norms
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)

            # Automatic clipping factor (no clamp)
            per_sample_clip_factor = self.max_grad_norm / (per_sample_norms + 0.01)
        else:
            # Empty case: produce an empty tensor on a sensible device
            device = (
                self.params[0].device if len(self.params) > 0 else torch.device("cpu")
            )
            per_sample_clip_factor = torch.tensor([], device=device)

        for p in self.params:
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)

            # cast per-sample gradients to optimizer parameter dtype (e.g., fp32)
            grad_sample = grad_sample.to(p.dtype)

            clip_factor_on_device = per_sample_clip_factor.to(grad_sample.device).to(
                p.dtype
            )
            grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)


class DPPerLayerAutomaticClippingOptimizer(DPPerLayerOptimizer):
    """
    Per-layer variant of automatic clipping.

    For each parameter (layer), we compute the per-sample clip factor using the
    corresponding per-layer ``max_grad_norm``::

        per_sample_clip_factor = max_grad_norm / (per_sample_norms + 0.01)

    This allows each layer to have different clipping behavior based on its own
    gradient magnitude distribution, which can improve training stability and
    utility compared to global clipping.

    This approach is described in:
    *"Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger"*
    https://arxiv.org/pdf/2206.07136

    Note:
        This optimizer is automatically instantiated when using
        ``PrivacyEngine.make_private()`` with ``clipping="automatic_per_layer"``.

    See Also:
        - :class:`~opacus.optimizers.perlayeroptimizer.DPPerLayerOptimizer`: Base per-layer DP optimizer
        - :class:`~DPAutomaticClippingOptimizer`: All-layer variant of automatic clipping
        - :class:`~opacus.optimizers.ddpoptimizer_automatic_clipping.DistributedDPPerLayerAutomaticClippingOptimizer`: Distributed version
    """

    def clip_and_accumulate(self):
        for p, max_grad_norm in zip(self.params, self.max_grad_norms):
            _check_processed_flag(p.grad_sample)

            grad_sample = self._get_flat_grad_sample(p)
            # per-sample norms for this parameter (collapse parameter dims)
            per_sample_norms = grad_sample.norm(
                2, dim=tuple(range(1, grad_sample.ndim))
            )

            per_sample_clip_factor = max_grad_norm / (per_sample_norms + 0.01)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)


__all__ = [
    "DPAutomaticClippingOptimizer",
    "DPPerLayerAutomaticClippingOptimizer",
]
