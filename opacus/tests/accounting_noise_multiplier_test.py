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
Tests for accounting_noise_multiplier property to ensure correct privacy accounting,
especially for AdaClipDPOptimizer which internally adjusts noise_multiplier.
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


class AccountingNoiseMultiplierTest(unittest.TestCase):
    """Test that accounting_noise_multiplier property works correctly."""

    def setUp(self):
        # For AdaClip: noise_multiplier must be < 2 * unclipped_num_std
        self.noise_multiplier = 1.5
        self.unclipped_num_std = 1.0  # 2 * 1.0 = 2.0 > 1.5, so valid
        self.max_grad_norm = 1.0
        torch.manual_seed(42)

    def test_dpoptimizer_accounting_noise_multiplier(self):
        """Test that DPOptimizer.accounting_noise_multiplier returns noise_multiplier."""
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
            expected_batch_size=32,
        )

        # For standard DPOptimizer, accounting_noise_multiplier should equal noise_multiplier
        self.assertEqual(
            dp_optimizer.accounting_noise_multiplier,
            dp_optimizer.noise_multiplier,
        )
        self.assertEqual(dp_optimizer.accounting_noise_multiplier, self.noise_multiplier)

    def test_adaclip_stores_original_noise_multiplier(self):
        """Test that AdaClipDPOptimizer stores and returns the original noise_multiplier."""
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

        # Store the adjusted noise_multiplier for comparison
        adjusted_noise_multiplier = adaclip_optimizer.noise_multiplier

        # accounting_noise_multiplier should return the original value
        self.assertEqual(
            adaclip_optimizer.accounting_noise_multiplier, self.noise_multiplier
        )

        # Verify that noise_multiplier was adjusted according to Theorem 1
        # noise_multiplier = (sigma^-2 - (2*sigma_u)^-2)^(-1/2)
        expected_adjusted = (
            self.noise_multiplier ** (-2) - (2 * self.unclipped_num_std) ** (-2)
        ) ** (-1 / 2)
        self.assertAlmostEqual(
            adjusted_noise_multiplier, expected_adjusted, places=5
        )

        # accounting_noise_multiplier should differ from the adjusted noise_multiplier
        self.assertNotEqual(
            adaclip_optimizer.accounting_noise_multiplier,
            adaclip_optimizer.noise_multiplier,
        )

    def test_adaclip_accounting_with_zero_noise(self):
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

        # Both should be zero
        self.assertEqual(adaclip_optimizer.accounting_noise_multiplier, 0.0)
        self.assertEqual(adaclip_optimizer.noise_multiplier, 0.0)

    def test_accountant_uses_accounting_noise_multiplier(self):
        """Test that accountant hook code path uses accounting_noise_multiplier from optimizer."""
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

        # Manually call accountant.step with accounting_noise_multiplier
        # (mimicking what the hook would do)
        initial_len = len(accountant)
        accountant.step(
            noise_multiplier=adaclip_optimizer.accounting_noise_multiplier,
            sample_rate=sample_rate,
        )

        # Accountant should have recorded one step
        self.assertEqual(len(accountant), initial_len + 1)

        # Verify that the accountant used the original noise_multiplier
        # accountant.history stores tuples of (noise_multiplier, sample_rate, num_steps)
        last_entry = accountant.history[-1]
        recorded_noise_multiplier = last_entry[0]

        # Should use accounting_noise_multiplier (original), not adjusted
        self.assertEqual(recorded_noise_multiplier, self.noise_multiplier)
        self.assertNotEqual(recorded_noise_multiplier, adaclip_optimizer.noise_multiplier)

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

        # Verify accountant recorded steps with accounting_noise_multiplier
        self.assertGreater(len(accountant), 0)

        # All recorded noise multipliers should be the original value
        for entry in accountant.history:
            recorded_noise = entry[0]
            # Should match accounting_noise_multiplier (original)
            self.assertEqual(recorded_noise, dp_optimizer.accounting_noise_multiplier)
            # Should NOT match the adjusted noise_multiplier
            self.assertNotEqual(recorded_noise, dp_optimizer.noise_multiplier)

    def test_adaclip_accounting_multiplier_immutable(self):
        """Test that accounting_noise_multiplier remains constant even as noise_multiplier changes."""
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

        # Store original values
        original_accounting = adaclip_optimizer.accounting_noise_multiplier
        original_noise = adaclip_optimizer.noise_multiplier

        # Manually modify noise_multiplier (simulating what might happen during training)
        adaclip_optimizer.noise_multiplier = 2.0

        # accounting_noise_multiplier should remain unchanged
        self.assertEqual(adaclip_optimizer.accounting_noise_multiplier, original_accounting)
        self.assertEqual(adaclip_optimizer.accounting_noise_multiplier, self.noise_multiplier)

        # But noise_multiplier should reflect the change
        self.assertNotEqual(adaclip_optimizer.noise_multiplier, original_noise)
        self.assertEqual(adaclip_optimizer.noise_multiplier, 2.0)

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

        # Both should report the same accounting_noise_multiplier
        self.assertEqual(
            dp_optimizer.accounting_noise_multiplier,
            adaclip_optimizer.accounting_noise_multiplier,
        )

        # But their actual noise_multiplier values differ
        self.assertNotEqual(
            dp_optimizer.noise_multiplier,
            adaclip_optimizer.noise_multiplier,
        )


if __name__ == "__main__":
    unittest.main()
