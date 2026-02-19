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

import copy
import logging
from typing import Callable, Optional

import torch
from torch.optim import Optimizer

from .optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)


logger = logging.getLogger(__name__)
logger.disabled = True


class DPOptimizerFastGradientClipping(DPOptimizer):
    """
    ``torch.optim.Optimizer`` wrapper to implement Fast Gradient and Ghost Clipping -- modifies DPOptimizer
    to only add noise to the average gradient, without clipping.

    Can be used with any ``torch.optim.Optimizer`` subclass as an underlying optimizer.
    ``DPOptimizerFastGradientClipping`` assumes that parameters over which it performs optimization belong
    to GradSampleModuleFastGradientClipping and therefore have the ``grad_sample`` attribute.

    On a high level ``DPOptimizerFastGradientClipping``'s step looks like this:
    1) Add Gaussian noise to ``p.grad`` calibrated to a given noise multiplier and
    max grad norm limit (``std = noise_multiplier * max_grad_norm``).
    2) Call underlying optimizer to perform optimization step

    Examples:
        >>> module = MyCustomModel()
        >>> optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
        >>> dp_optimizer = DPOptimizerFastGradientClipping(
        ...     optimizer=optimizer,
        ...     noise_multiplier=1.0,
        ...     max_grad_norm=1.0,
        ...     expected_batch_size=4,
        ... )
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
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for calculating the standard devition of noise added
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """

        super().__init__(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            expected_batch_size=expected_batch_size,
            max_grad_norm=max_grad_norm,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
            **kwargs,
        )

    @property
    def accumulated_iterations(self) -> int:
        """
        Returns number of batches currently accumulated and not yet processed.

        In other words ``accumulated_iterations`` tracks the number of forward/backward
        passed done in between two optimizer steps. The value would typically be 1,
        but there are possible exceptions.

        Used by privacy accountants to calculate real sampling rate.
        """
        return 1

    def accumulate(self):
        """
        Performs gradient accumulation.
        Stores aggregated gradients into `p.summed_grad```
        """
        for p in self.params:
            if p.summed_grad is not None:
                p.summed_grad.add_(p.grad.data)
            else:
                p.summed_grad = copy.deepcopy(p.grad.data)

    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients.

        Clears ``p.grad`` and ``p.summed_grad`` for all of it's parameters

        Notes:
            ``set_to_none`` argument only affects ``p.grad`` and
            ``p.summed_grad`` is never zeroed out and always set to None.
            Normal grads can do this, because their shape is always the same.
            Grad samples do not behave like this, as we accumulate gradients from different
            batches in a list

        Args:
            set_to_none: instead of setting to zero, set the grads to None. (only
            affects regular gradients. Per sample gradients are always set to None)
        """

        if set_to_none is False:
            logger.debug(
                "Despite set_to_none is set to False, "
                "opacus will set p.summed_grad to None due to "
                "non-trivial gradient accumulation behaviour"
            )

        for p in self.params:
            if not self._is_last_step_skipped:
                p.summed_grad = None
        self.original_optimizer.zero_grad(set_to_none)

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """
        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer

        self.accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True

    def _get_flat_grad_sample(self, p: torch.Tensor):
        """
        Redefines a parent class' function to not do anything
        """
        pass

    def clip_and_accumulate(self):
        """
        Redefines a parent class' function to not do anything
        """
        pass

    def add_noise(self):
        """
        Adds noise to gradients. Stores noised result in ``p.grad``.

        Uses ``_adjusted_noise_multiplier`` if set (for adaptive clipping),
        otherwise falls back to ``noise_multiplier``. This separation ensures
        correct privacy accounting when using adaptive clipping, where the
        original noise_multiplier is preserved for accounting while the
        adjusted value is used only for noise generation.
        """
        # Use adjusted noise multiplier if set (for adaptive clipping),
        # otherwise use the standard noise_multiplier
        effective_noise_multiplier = getattr(
            self, "_adjusted_noise_multiplier", self.noise_multiplier
        )

        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = _generate_noise(
                std=effective_noise_multiplier * self.max_grad_norm,
                reference=p.summed_grad,
                generator=self.generator,
                secure_mode=self.secure_mode,
            )
            p.grad = (p.summed_grad + noise).view_as(p)

            _mark_as_processed(p.summed_grad)
