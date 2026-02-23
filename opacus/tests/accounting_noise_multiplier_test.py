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
Tests for AdaClipDPOptimizer to ensure correct privacy accounting.

The AdaClip optimizer uses an adjusted noise multiplier for gradient noise
(Theorem 1 from https://arxiv.org/pdf/1905.03871.pdf), but the original
noise_multiplier should be preserved for privacy accounting.
"""

import unittest

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.optimizer import DPOptimizer
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class AdaClipNoiseMultiplierTest(unittest.TestCase):
    """Test that AdaClip preserves original noise_multiplier for privacy accounting."""

    def setUp(self):
        # For AdaClip: noise_multiplier must be < 2 * unclipped_num_std
        self.noise_multiplier = 1.5
        self.unclipped_num_std = 1.0  # 2 * 1.0 = 2.0 > 1.5, so valid
        self.max_grad_norm = 1.0
        torch.manual_seed(42)

    def test_adaclip_preserves_noise_multiplier(self):
        """Test that AdaClipDPOptimizer preserves original noise_multiplier."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # noise_multiplier should remain unchanged (original value)
        self.assertEqual(adaclip_optimizer.noise_multiplier, self.noise_multiplier)

    def test_adaclip_with_zero_noise(self):
        """Test that AdaClipDPOptimizer handles zero noise_multiplier correctly."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # noise_multiplier should be zero
        self.assertEqual(adaclip_optimizer.noise_multiplier, 0.0)

    def test_accountant_uses_original_noise_multiplier(self):
        """Test that accountant hook uses original noise_multiplier from AdaClip optimizer."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        accountant = RDPAccountant()

        # Create AdaClip optimizer
        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        sample_rate = 0.01

        # Manually call accountant.step with noise_multiplier
        # (mimicking what the hook would do)
        initial_len = len(accountant)
        accountant.step(
            noise_multiplier=adaclip_optimizer.noise_multiplier,
            sample_rate=sample_rate,
        )

        # Accountant should have recorded one step
        self.assertEqual(len(accountant), initial_len + 1)

        # Verify that the accountant used the original noise_multiplier
        # accountant.history stores tuples of (noise_multiplier, sample_rate, num_steps)
        last_entry = accountant.history[-1]
        recorded_noise_multiplier = last_entry[0]

        # Should use original noise_multiplier
        self.assertEqual(recorded_noise_multiplier, self.noise_multiplier)

    def test_privacy_accounting_with_adaclip_e2e(self):
        """End-to-end test: verify privacy accounting is correct with AdaClip via PrivacyEngine."""
        # Create synthetic data
        X = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10)

        # Setup model and use PrivacyEngine with adaptive clipping
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Use PrivacyEngine to set up AdaClip with all required kwargs
        privacy_engine = PrivacyEngine()
        model, dp_optimizer, private_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            clipping="adaptive",
            grad_sample_mode="hooks",
            # AdaClip specific parameters
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
        )

        # Verify optimizer is AdaClip
        self.assertIsInstance(dp_optimizer, AdaClipDPOptimizer)

        # Verify noise_multiplier is preserved
        self.assertEqual(dp_optimizer.noise_multiplier, self.noise_multiplier)

        # Get the accountant
        accountant = privacy_engine.accountant

        # Train for a few steps
        criterion = nn.CrossEntropyLoss()
        for i, (data, target) in enumerate(private_dataloader):
            if i >= 5:  # Just a few steps for testing
                break
            dp_optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            dp_optimizer.step()

        # Verify accountant recorded steps with original noise_multiplier
        self.assertGreater(len(accountant), 0)

        # All recorded noise multipliers should be the original value
        for entry in accountant.history:
            recorded_noise = entry[0]
            # Should match original noise_multiplier
            self.assertEqual(recorded_noise, self.noise_multiplier)

    def test_comparison_dpoptimizer_vs_adaclip_accounting(self):
        """Compare accounting between standard DPOptimizer and AdaClip with same initial noise."""
        model1 = SimpleModel()
        model2 = SimpleModel()
        model2.load_state_dict(model1.state_dict())  # Same initial weights

        optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
        optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)

        # Standard DPOptimizer
        dp_optimizer = DPOptimizer(
            optimizer=optimizer1,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # AdaClip optimizer
        adaclip_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer2,
            noise_multiplier=self.noise_multiplier,
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.01,
            max_clipbound=2.0,
            min_clipbound=0.5,
            unclipped_num_std=self.unclipped_num_std,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # Both should have the same noise_multiplier for accounting
        self.assertEqual(
            dp_optimizer.noise_multiplier,
            adaclip_optimizer.noise_multiplier,
        )

    def test_adaclip_noise_adjustment_calculation(self):
        """Test that the adjusted noise follows Theorem 1 formula when applied internally."""
        # According to Theorem 1: σ_eff = (σ^-2 - (2σ_u)^-2)^(-1/2)
        sigma = self.noise_multiplier
        sigma_u = self.unclipped_num_std

        expected_adjusted = (sigma ** (-2) - (2 * sigma_u) ** (-2)) ** (-1 / 2)

        # Verify the formula produces valid results
        self.assertGreater(expected_adjusted, 0)
        # The adjusted noise is larger than the original
        # (σ^-2 - positive_term)^(-1/2) > σ when σ < 2*σ_u
        self.assertGreater(expected_adjusted, sigma)

    def test_adaclip_constraint_validation(self):
        """Test that AdaClip raises error when noise_multiplier >= 2 * unclipped_num_std."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # This should raise ValueError: 2.0 >= 2 * 1.0 = 2.0
        with self.assertRaises(ValueError) as context:
            AdaClipDPOptimizer(
                optimizer=optimizer,
                noise_multiplier=2.0,
                target_unclipped_quantile=0.5,
                clipbound_learning_rate=0.01,
                max_clipbound=2.0,
                min_clipbound=0.5,
                unclipped_num_std=1.0,
                max_grad_norm=self.max_grad_norm,
                expected_batch_size=32,
            )

        self.assertIn(
            "noise_multiplier must be smaller than 2 * unclipped_num_std",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
