#!/usr/bin/env python3
# Copyright (c) 2026 SlaClip authors.
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

"""PrivacyEngine entry point for the SlaClip research prototype."""

from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
from opacus.optimizers import DPOptimizer
from opacus.privacy_engine import PrivacyEngine
from torch import optim

from .slaclipoptimizer import SlaClipDPOptimizer


class SlaClipPrivacyEngine(PrivacyEngine):
    """Prepare Opacus training objects with :class:`SlaClipDPOptimizer`.

    Passing no controller enables the indicator-only mode. Passing the paper's
    ``SlaClipController`` enables the complete SlaClip adaptation rule.
    """

    def __init__(
        self,
        *,
        accountant: str = "prv",
        secure_mode: bool = False,
        num_slots: Optional[int] = None,
        clipping_controller: Optional[Callable[[float, torch.Tensor], float]] = None,
    ):
        super().__init__(accountant=accountant, secure_mode=secure_mode)
        self.num_slots = num_slots
        self.clipping_controller = clipping_controller

    def _prepare_optimizer(
        self,
        *,
        optimizer: optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        **kwargs,
    ) -> SlaClipDPOptimizer:
        if distributed:
            raise ValueError("SlaClip does not currently support distributed training")
        if clipping != "flat":
            raise ValueError("SlaClip requires clipping='flat'")
        if "ghost" in grad_sample_mode:
            raise ValueError("SlaClip does not currently support ghost clipping")
        if isinstance(max_grad_norm, list):
            raise ValueError("SlaClip requires a scalar max_grad_norm")

        if isinstance(optimizer, DPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        return SlaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=float(max_grad_norm),
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=self.secure_mode,
            num_slots=self.num_slots,
            clipping_controller=self.clipping_controller,
            **kwargs,
        )
