"""Experimental SlaClip optimizer implementation for research use.

This self-contained implementation is stored under ``research/slaclip`` and
does not modify Opacus core routing or public APIs.
"""

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

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
from opacus.optimizers.optimizer import (
    DPOptimizer,
    _check_processed_flag,
    _generate_noise,
    _mark_as_processed,
)
from torch.optim import Optimizer


def paper_recommended_k(batch_size: int, noise_multiplier: float = 1.0) -> int:
    """Select the largest integer ``K`` satisfying SlaClip equation (36)."""

    z_0995 = 2.576

    batch_size = int(batch_size)
    noise_multiplier = float(noise_multiplier)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")

    k_max = (batch_size / (2.0 * z_0995 * noise_multiplier)) ** (2.0 / 3.0)
    return max(1, math.floor(k_max))


class SlaClipController:
    """Paper-faithful clipping update from SlaClip equations (28)-(30)."""

    def __init__(
        self,
        *,
        eta: float = 0.5,
        min_clipbound: float = 0.1,
        max_clipbound: float = 50.0,
    ):
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError("eta must be finite and positive")
        if not math.isfinite(min_clipbound) or min_clipbound <= 0:
            raise ValueError("min_clipbound must be finite and positive")
        if not math.isfinite(max_clipbound) or max_clipbound <= min_clipbound:
            raise ValueError(
                "max_clipbound must be finite and larger than min_clipbound"
            )
        self.eta = float(eta)
        self.min_clipbound = float(min_clipbound)
        self.max_clipbound = float(max_clipbound)

    def __call__(self, current_clip: float, slack_indicator: torch.Tensor) -> float:
        if not math.isfinite(current_clip) or current_clip <= 0:
            raise ValueError("current_clip must be finite and positive")
        if slack_indicator.ndim != 1 or slack_indicator.numel() == 0:
            raise ValueError("slack_indicator must be a non-empty vector")
        if not bool(torch.isfinite(slack_indicator).all().item()):
            raise ValueError("slack_indicator must contain only finite values")

        near_threshold = float(slack_indicator[0].item())
        near_zero = float(slack_indicator[-1].item())
        near_zero_adjusted = max(0.0, min(1.0, near_zero / current_clip))
        target = max(0.0, min(1.0, 1.0 - (1.0 - near_zero_adjusted) / 2.0))
        log_next_clip = math.log(current_clip) + self.eta * (target - near_threshold)
        if log_next_clip <= math.log(self.min_clipbound):
            return self.min_clipbound
        if log_next_clip >= math.log(self.max_clipbound):
            return self.max_clipbound
        return float(math.exp(log_next_clip))


class SlaClipDPOptimizer(DPOptimizer):
    """DPOptimizer that jointly releases a private Slack Indicator.

    The optimizer implements SlaClip equations (6)-(11) without materializing a
    ``d + K`` tensor. Clipped gradients and slack coordinates are accumulated
    separately, then perturbed with independent coordinates of the same Gaussian
    mechanism. Passing a ``clipping_controller`` additionally applies step 2 of
    SlaClip; leaving it as ``None`` exposes only the private CDF information.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        num_slots: Optional[int] = None,
        clipping_controller: Optional[Callable[[float, torch.Tensor], float]] = None,
    ):
        if expected_batch_size is None or expected_batch_size <= 0:
            raise ValueError("expected_batch_size must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )

        self.K = (
            paper_recommended_k(expected_batch_size, noise_multiplier)
            if num_slots is None
            else int(num_slots)
        )
        if self.K <= 0:
            raise ValueError("num_slots must be a positive integer")

        self.clipping_controller = clipping_controller
        self._lambda_t = 0.0
        self._slack_sum: Optional[torch.Tensor] = None
        self._slack_indicator: Optional[torch.Tensor] = None

    @property
    def current_clip(self) -> float:
        """Current clipping threshold."""

        return float(self.max_grad_norm)

    @property
    def slack_indicator(self) -> torch.Tensor:
        """Most recently released noisy, normalized Slack Indicator.

        This property never exposes the unnoised per-sample slack or aggregate.
        """

        if self._slack_indicator is None:
            raise RuntimeError(
                "Slack Indicator is available only after optimizer.step()"
            )
        return self._slack_indicator.detach().clone()

    def zero_grad(self, set_to_none: bool = False):
        super().zero_grad(set_to_none)
        if not self._is_last_step_skipped:
            self._slack_sum = None
            self._lambda_t = 0.0

    def _release_denom(self) -> float:
        denom = float(self.expected_batch_size) * float(self.accumulated_iterations)
        if denom <= 0:
            raise ValueError("Expected release denominator must be positive")
        return denom

    def _encode_slack(
        self, per_sample_norms: torch.Tensor, current_clip: float
    ) -> torch.Tensor:
        """Encode SlaClip equations (7)-(8) for a batch of gradient norms."""

        lambda_t = float(current_clip / math.sqrt(self.K))
        scaled_slack = torch.clamp(
            current_clip - per_sample_norms, min=0.0
        ) * math.sqrt(self.K)
        full_slots = torch.floor(scaled_slack / lambda_t).to(torch.int64)
        full_slots = torch.clamp(full_slots, min=0, max=self.K)
        residual = scaled_slack - full_slots.to(scaled_slack.dtype) * lambda_t
        residual = torch.where(
            full_slots >= self.K, torch.zeros_like(residual), residual
        )

        slot_index = torch.arange(self.K, device=per_sample_norms.device).view(
            1, self.K
        )
        slack = (slot_index < full_slots.view(-1, 1)).to(
            dtype=per_sample_norms.dtype
        ) * float(lambda_t)

        has_residual = full_slots < self.K
        if has_residual.any():
            residual_slot = torch.clamp(full_slots, max=self.K - 1)
            slack[has_residual, residual_slot[has_residual]] = residual[has_residual]
        return slack

    def clip_and_accumulate(self):
        """Clip gradients as DPOptimizer and also aggregate encoded slack."""

        grad_samples = self.grad_samples
        if not grad_samples:
            return

        if len(grad_samples[0]) == 0:
            per_sample_norms = torch.zeros(
                (0,), device=grad_samples[0].device, dtype=grad_samples[0].dtype
            )
            per_sample_clip_factor = per_sample_norms
        else:
            per_param_norms = [
                grad_sample.reshape(len(grad_sample), -1).norm(2, dim=-1)
                for grad_sample in grad_samples
            ]
            target_device = per_param_norms[0].device
            per_param_norms = [norm.to(target_device) for norm in per_param_norms]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.current_clip / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for parameter in self.params:
            _check_processed_flag(parameter.grad_sample)
            grad_sample = self._get_flat_grad_sample(parameter).to(parameter.dtype)
            clip_factor = per_sample_clip_factor.to(
                device=grad_sample.device, dtype=parameter.dtype
            )
            grad = torch.einsum("i,i...", clip_factor, grad_sample)
            if parameter.summed_grad is None:
                parameter.summed_grad = grad
            else:
                parameter.summed_grad += grad
            _mark_as_processed(parameter.grad_sample)

        lambda_t = float(self.current_clip / math.sqrt(self.K))
        if self._lambda_t and not math.isclose(
            self._lambda_t, lambda_t, rel_tol=1e-12, abs_tol=0.0
        ):
            raise ValueError(
                "Clipping threshold changed while accumulating a logical batch"
            )
        self._lambda_t = lambda_t

        batch_slack_sum = self._encode_slack(per_sample_norms, self.current_clip).sum(
            dim=0
        )
        if self._slack_sum is None:
            self._slack_sum = batch_slack_sum
        else:
            self._slack_sum += batch_slack_sum.to(self._slack_sum.device)

    def add_noise(self):
        """Release gradient and Slack Indicator as one extended mechanism."""

        current_clip = self.current_clip

        # Keep the first d coordinates on the native DPOptimizer path. Drawing
        # the K slack coordinates afterwards is distributionally identical to
        # one isotropic Gaussian draw in d + K dimensions.
        super().add_noise()

        if self._slack_sum is None or self._lambda_t <= 0:
            raise RuntimeError("Slack must be accumulated before adding noise")
        slack_noise = _generate_noise(
            std=self.noise_multiplier * current_clip,
            reference=self._slack_sum,
            generator=self.generator,
            secure_mode=self.secure_mode,
        )
        self._slack_indicator = (self._slack_sum + slack_noise) / (
            self._lambda_t * self._release_denom()
        )

        if self.clipping_controller is not None:
            next_clip = float(
                self.clipping_controller(current_clip, self._slack_indicator)
            )
            if not math.isfinite(next_clip) or next_clip <= 0:
                raise ValueError(
                    "clipping_controller must return a finite positive value"
                )
            self.max_grad_norm = next_clip


SlaClipOptimizer = SlaClipDPOptimizer
