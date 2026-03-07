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

"""
Regression test for GitHub issue #792:
  Fast gradient clipping ignores masking (ignore_index)
  https://github.com/meta-pytorch/opacus/issues/792

When DPLossFastGradientClipping computes per-sample loss for NLP tasks
(using the `shape` parameter), the mean reduction uses .mean(dim=1) which
divides by the full sequence length. This ignores the `ignore_index` setting
from the criterion, leading to an incorrectly diluted loss when many tokens
are masked (e.g., SQuAD tasks where only a few tokens are the answer).
"""

import unittest

import torch
import torch.nn as nn
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping


class GithubIssueRegressionTest(unittest.TestCase):
    """Regression tests for GitHub issues."""

    def test_fast_gradient_clipping_respects_ignore_index(self):
        """
        GitHub issue #792: DPLossFastGradientClipping should respect
        ignore_index when computing mean reduction for NLP tasks.

        When criterion has ignore_index set, the mean reduction in
        DPLossFastGradientClipping.__call__ should only average over
        non-ignored positions, matching PyTorch's CrossEntropyLoss behavior.
        """
        batch_size = 2
        seq_len = 10
        vocab_size = 5
        ignore_index = -100

        # Create a simple linear model (logits producer)
        model = nn.Linear(8, vocab_size)
        gsm = GradSampleModuleFastGradientClipping(
            model,
            max_grad_norm=1.0,
            use_ghost_clipping=True,
            loss_reduction="mean",
        )
        optimizer = torch.optim.SGD(gsm.parameters(), lr=0.1)
        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=batch_size,
            loss_reduction="mean",
        )

        # Criterion with ignore_index
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

        dp_criterion = DPLossFastGradientClipping(
            gsm, dp_optimizer, criterion, loss_reduction="mean"
        )

        # Create input: [batch_size, seq_len, hidden_dim]
        inputs = torch.randn(batch_size, seq_len, 8)
        # Logits: [batch_size, seq_len, vocab_size]
        logits = gsm(inputs)

        # Create targets where most tokens are masked (ignore_index)
        # Only 2 out of 10 tokens per sample are real targets
        targets = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long)
        targets[0, 0] = 1  # Sample 0: only position 0 is real
        targets[0, 3] = 2  # Sample 0: only position 3 is real
        targets[1, 1] = 0  # Sample 1: only position 1 is real
        targets[1, 5] = 3  # Sample 1: only position 5 is real

        # Flatten for CrossEntropyLoss: logits [B*T, V], targets [B*T]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)

        # Compute DP loss with shape parameter (NLP path)
        shape = (batch_size, seq_len, vocab_size)
        dp_loss = dp_criterion(flat_logits, flat_targets, shape=shape)

        # Compute reference per-sample loss manually using PyTorch's
        # CrossEntropyLoss with ignore_index and reduction="none"
        ref_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        ref_loss_flat = ref_criterion(flat_logits.detach(), flat_targets)
        ref_loss_per_sample = ref_loss_flat.view(batch_size, seq_len)

        # Manual mean that respects ignore_index:
        # only average over non-ignored positions per sample
        mask = flat_targets.view(batch_size, seq_len) != ignore_index
        ref_mean_per_sample = (ref_loss_per_sample * mask).sum(dim=1) / mask.sum(dim=1)

        # The DP loss per-sample should match the reference (which respects ignore_index)
        self.assertEqual(dp_loss.loss_per_sample.shape, ref_mean_per_sample.shape)
        self.assertTrue(
            torch.allclose(dp_loss.loss_per_sample, ref_mean_per_sample, atol=1e-5),
            f"DPLossFastGradientClipping does not respect ignore_index.\n"
            f"  DP loss per sample: {dp_loss.loss_per_sample}\n"
            f"  Expected (respecting ignore_index): {ref_mean_per_sample}\n"
            f"  Naive mean (ignoring masking): {ref_loss_per_sample.mean(dim=1)}\n"
            f"  The DP loss matches the naive mean, meaning ignore_index is not respected.",
        )

    def test_fast_gradient_clipping_sum_reduction_with_ignore_index(self):
        """
        Same as above but for sum reduction — sum should also only sum
        over non-ignored positions (matching PyTorch behavior).
        """
        batch_size = 2
        seq_len = 10
        vocab_size = 5
        ignore_index = -100

        model = nn.Linear(8, vocab_size)
        gsm = GradSampleModuleFastGradientClipping(
            model,
            max_grad_norm=1.0,
            use_ghost_clipping=True,
            loss_reduction="sum",
        )
        optimizer = torch.optim.SGD(gsm.parameters(), lr=0.1)
        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=batch_size,
            loss_reduction="sum",
        )

        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
        dp_criterion = DPLossFastGradientClipping(
            gsm, dp_optimizer, criterion, loss_reduction="sum"
        )

        inputs = torch.randn(batch_size, seq_len, 8)
        logits = gsm(inputs)

        targets = torch.full((batch_size, seq_len), ignore_index, dtype=torch.long)
        targets[0, 0] = 1
        targets[0, 3] = 2
        targets[1, 1] = 0
        targets[1, 5] = 3

        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        shape = (batch_size, seq_len, vocab_size)

        dp_loss = dp_criterion(flat_logits, flat_targets, shape=shape)

        # Reference: sum reduction already works correctly since
        # CrossEntropyLoss(reduction="none") returns 0 for ignored positions
        ref_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="none")
        ref_loss_flat = ref_criterion(flat_logits.detach(), flat_targets)
        ref_loss_per_sample = ref_loss_flat.view(batch_size, seq_len)
        ref_sum_per_sample = ref_loss_per_sample.sum(dim=1)

        self.assertTrue(
            torch.allclose(dp_loss.loss_per_sample, ref_sum_per_sample, atol=1e-5),
            f"Sum reduction with ignore_index mismatch.\n"
            f"  DP loss: {dp_loss.loss_per_sample}\n"
            f"  Expected: {ref_sum_per_sample}",
        )

    def test_fast_gradient_clipping_no_ignore_index_unchanged(self):
        """
        Ensure that when no ignore_index is set (or no tokens are masked),
        behavior is unchanged — plain .mean(dim=1) should still work.
        """
        batch_size = 2
        seq_len = 5
        vocab_size = 4

        model = nn.Linear(8, vocab_size)
        gsm = GradSampleModuleFastGradientClipping(
            model,
            max_grad_norm=1.0,
            use_ghost_clipping=True,
            loss_reduction="mean",
        )
        optimizer = torch.optim.SGD(gsm.parameters(), lr=0.1)
        dp_optimizer = DPOptimizerFastGradientClipping(
            optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=batch_size,
            loss_reduction="mean",
        )

        # No ignore_index set
        criterion = nn.CrossEntropyLoss(reduction="mean")
        dp_criterion = DPLossFastGradientClipping(
            gsm, dp_optimizer, criterion, loss_reduction="mean"
        )

        inputs = torch.randn(batch_size, seq_len, 8)
        logits = gsm(inputs)

        # All tokens are valid
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        shape = (batch_size, seq_len, vocab_size)

        dp_loss = dp_criterion(flat_logits, flat_targets, shape=shape)

        # Reference with no masking: plain mean should be the same
        ref_criterion = nn.CrossEntropyLoss(reduction="none")
        ref_loss_flat = ref_criterion(flat_logits.detach(), flat_targets)
        ref_mean_per_sample = ref_loss_flat.view(batch_size, seq_len).mean(dim=1)

        self.assertTrue(
            torch.allclose(dp_loss.loss_per_sample, ref_mean_per_sample, atol=1e-5),
            f"Behavior changed for non-masked case.\n"
            f"  DP loss: {dp_loss.loss_per_sample}\n"
            f"  Expected: {ref_mean_per_sample}",
        )
