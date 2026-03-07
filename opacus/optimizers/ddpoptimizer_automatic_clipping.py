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

from typing import Callable, List, Optional

import torch
from opacus.optimizers.optimizer_automatic_clipping import (
    DPAutomaticClippingOptimizer,
    DPPerLayerAutomaticClippingOptimizer,
)
from torch.optim import Optimizer


class DistributedDPAutomaticClippingOptimizer(DPAutomaticClippingOptimizer):
    """
    Distributed version of DPAutomaticClippingOptimizer for multi-GPU training.

    This optimizer extends :class:`~opacus.optimizers.optimizer_automatic_clipping.DPAutomaticClippingOptimizer`
    to work with PyTorch's distributed data parallel (DDP) training. It handles:

    - **Gradient Synchronization**: Uses ``all_reduce`` to sum gradients across all workers
    - **Coordinated Noise**: Only rank 0 generates noise to ensure consistency
    - **Proper Reduction**: Handles mean/sum loss reduction across workers

    The automatic clipping formula remains:
        ``per_sample_clip_factor = max_grad_norm / (per_sample_norms + 0.01)``

    As described in:
    *"Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger"*
    https://arxiv.org/pdf/2206.07136

    Args:
        optimizer: Wrapped optimizer instance
        noise_multiplier: Noise multiplier for differential privacy
        max_grad_norm: Maximum gradient norm for clipping
        expected_batch_size: Expected batch size (per worker)
        loss_reduction: How to reduce loss across workers ("mean" or "sum")
        generator: Random number generator for noise
        secure_mode: Whether to use secure random number generation

    Note:
        This optimizer is automatically instantiated when using
        ``PrivacyEngine.make_private()`` with ``distributed=True`` and ``clipping="automatic"``.

    See Also:
        - :class:`~opacus.optimizers.optimizer_automatic_clipping.DPAutomaticClippingOptimizer`: Non-distributed version
        - :class:`~DistributedDPPerLayerAutomaticClippingOptimizer`: Per-layer distributed variant
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        """Add noise only on rank 0, then broadcast to other workers."""
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def reduce_gradients(self):
        """Reduce gradients across all workers."""
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        """Perform optimization step with distributed gradient synchronization."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        else:
            return None


class DistributedDPPerLayerAutomaticClippingOptimizer(
    DPPerLayerAutomaticClippingOptimizer
):
    """
    Distributed per-layer automatic clipping optimizer for multi-GPU training.

    This optimizer extends :class:`~opacus.optimizers.optimizer_automatic_clipping.DPPerLayerAutomaticClippingOptimizer`
    to work with PyTorch's distributed data parallel (DDP) training.

    Combines the benefits of:
        - **Per-layer clipping**: Each layer has its own ``max_grad_norm``
        - **Automatic clipping**: Smooth scaling without hard clamping
        - **Distributed training**: Gradient synchronization across workers

    The per-layer automatic clipping formula is:
        ``per_sample_clip_factor = max_grad_norm[layer] / (per_sample_norms[layer] + 0.01)``

    As described in:
    *"Automatic Clipping: Differentially Private Deep Learning Made Easier and Stronger"*
    https://arxiv.org/pdf/2206.07136

    Args:
        optimizer: Wrapped optimizer instance
        noise_multiplier: Noise multiplier for differential privacy
        max_grad_norm: List of maximum gradient norms (one per parameter)
        expected_batch_size: Expected batch size (per worker)
        loss_reduction: How to reduce loss across workers ("mean" or "sum")
        generator: Random number generator for noise
        secure_mode: Whether to use secure random number generation

    Note:
        This optimizer is automatically instantiated when using
        ``PrivacyEngine.make_private()`` with ``distributed=True`` and
        ``clipping="automatic_per_layer"``.

    See Also:
        - :class:`~opacus.optimizers.optimizer_automatic_clipping.DPPerLayerAutomaticClippingOptimizer`: Non-distributed version
        - :class:`~DistributedDPAutomaticClippingOptimizer`: All-layer distributed variant
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: List[float],
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

    def add_noise(self):
        """Add noise only on rank 0, then broadcast to other workers."""
        if self.rank == 0:
            super().add_noise()
        else:
            for p in self.params:
                p.grad = p.summed_grad.view_as(p)

    def reduce_gradients(self):
        """Reduce gradients across all workers."""
        for p in self.params:
            if not p.requires_grad:
                continue
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            if self.loss_reduction == "mean":
                p.grad /= self.world_size

    def step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[torch.Tensor]:
        """Perform optimization step with distributed gradient synchronization."""
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            self.reduce_gradients()
            return self.original_optimizer.step()
        else:
            return None


__all__ = [
    "DistributedDPAutomaticClippingOptimizer",
    "DistributedDPPerLayerAutomaticClippingOptimizer",
]
