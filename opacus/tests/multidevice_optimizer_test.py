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
Tests for multi-device handling in DPOptimizer and AdaClipDPOptimizer.
These tests verify that optimizers correctly handle parameters and gradients
spread across multiple devices (e.g., when using device_map="auto" with accelerate).
"""

import unittest

import torch
import torch.nn as nn
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.optimizer import DPOptimizer


class MultiDeviceModel(nn.Module):
    """
    A simple model with parameters on different devices to simulate
    device_map="auto" behavior in multi-GPU setups.
    """

    def __init__(self, device1, device2):
        super().__init__()
        self.fc1 = nn.Linear(10, 20).to(device1)
        self.fc2 = nn.Linear(20, 5).to(device2)
        self.device1 = device1
        self.device2 = device2

    def forward(self, x):
        x = x.to(self.device1)
        x = torch.relu(self.fc1(x))
        x = x.to(self.device2)
        return self.fc2(x)


class MultiDeviceOptimizerTest(unittest.TestCase):
    """Test multi-device handling in optimizers."""

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dpoptimizer_multidevice_clip_and_accumulate(self):
        """Test that DPOptimizer handles parameters on different devices."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        model = MultiDeviceModel(device1, device2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Wrap optimizer with DPOptimizer
        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=4,
            loss_reduction="mean",
        )

        # Create batch and compute gradients
        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))
        criterion = nn.CrossEntropyLoss()

        # Simulate per-sample gradients on different devices
        model.zero_grad()
        for p in model.parameters():
            # Create fake per-sample gradients on the same device as the parameter
            p.grad_sample = torch.randn(batch_size, *p.shape, device=p.device)

        # This should not raise any device mismatch errors
        try:
            dp_optimizer.clip_and_accumulate()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in clip_and_accumulate: {e}")
            else:
                raise

        self.assertTrue(
            success, "clip_and_accumulate should handle multi-device parameters"
        )

        # Verify that summed_grad was created on the correct device for each parameter
        for p in model.parameters():
            self.assertIsNotNone(p.summed_grad, "summed_grad should be set")
            self.assertEqual(
                p.summed_grad.device,
                p.device,
                f"summed_grad should be on same device as parameter",
            )

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_adaclip_optimizer_multidevice_clip_and_accumulate(self):
        """Test that AdaClipDPOptimizer handles parameters on different devices."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        model = MultiDeviceModel(device1, device2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        # Wrap optimizer with AdaClipDPOptimizer
        dp_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            expected_batch_size=4,
            loss_reduction="mean",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.5,
        )

        # Create batch and compute gradients
        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))

        # Simulate per-sample gradients on different devices
        model.zero_grad()
        for p in model.parameters():
            # Create fake per-sample gradients on the same device as the parameter
            p.grad_sample = torch.randn(batch_size, *p.shape, device=p.device)

        # This should not raise any device mismatch errors
        try:
            dp_optimizer.clip_and_accumulate()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in clip_and_accumulate: {e}")
            else:
                raise

        self.assertTrue(
            success, "clip_and_accumulate should handle multi-device parameters"
        )

        # Verify that summed_grad was created on the correct device for each parameter
        for p in model.parameters():
            self.assertIsNotNone(p.summed_grad, "summed_grad should be set")
            self.assertEqual(
                p.summed_grad.device,
                p.device,
                f"summed_grad should be on same device as parameter",
            )

        # Verify AdaClip-specific tracking
        self.assertEqual(dp_optimizer.sample_size, batch_size)
        self.assertGreaterEqual(dp_optimizer.unclipped_num, 0)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dpoptimizer_multidevice_full_step(self):
        """Test full optimizer step with multi-device model."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        model = MultiDeviceModel(device1, device2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        dp_optimizer = DPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            expected_batch_size=4,
            loss_reduction="mean",
        )

        # Create batch
        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))
        criterion = nn.CrossEntropyLoss()

        # Simulate per-sample gradients
        model.zero_grad()
        for p in model.parameters():
            p.grad_sample = torch.randn(batch_size, *p.shape, device=p.device)

        # Full step should work without device errors
        try:
            result = dp_optimizer.step()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in step: {e}")
            else:
                raise

        self.assertTrue(success, "step should handle multi-device parameters")

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_adaclip_optimizer_multidevice_full_step(self):
        """Test full optimizer step with AdaClip and multi-device model."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        model = MultiDeviceModel(device1, device2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        dp_optimizer = AdaClipDPOptimizer(
            optimizer=optimizer,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            expected_batch_size=4,
            loss_reduction="mean",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=10.0,
            min_clipbound=0.01,
            unclipped_num_std=0.5,
        )

        # Create batch
        batch_size = 4
        x = torch.randn(batch_size, 10)
        y = torch.randint(0, 5, (batch_size,))

        # Simulate per-sample gradients
        model.zero_grad()
        for p in model.parameters():
            p.grad_sample = torch.randn(batch_size, *p.shape, device=p.device)

        # Full step should work without device errors
        try:
            result = dp_optimizer.step()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in step: {e}")
            else:
                raise

        self.assertTrue(success, "step should handle multi-device parameters")

        # Verify clipbound was updated
        self.assertIsNotNone(dp_optimizer.max_grad_norm)


if __name__ == "__main__":
    unittest.main()
