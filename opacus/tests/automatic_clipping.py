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
Tests for automatic clipping optimizers in single-GPU/non-distributed setting.

For distributed (multi-GPU) tests, see multigpu_automatic_clipping_test.py.
"""

import unittest

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.optimizers.optimizer_automatic_clipping import (
    DPAutomaticClippingOptimizer,
    DPPerLayerAutomaticClippingOptimizer,
)
from torch.utils.data import DataLoader, TensorDataset


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class AutomaticClippingTest(unittest.TestCase):
    def setUp(self):
        self.DATA_SIZE = 64
        self.BATCH_SIZE = 16
        self.LR = 0.5
        self.NOISE_MULTIPLIER = 0.0  # No noise for deterministic tests
        self.MAX_GRAD_NORM = 1.0

    def _init_data(self):
        data = torch.randn(self.DATA_SIZE, 1, 28, 28)
        labels = torch.randint(0, 10, (self.DATA_SIZE,))
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=self.BATCH_SIZE)

    def test_automatic_clipping_basic(self):
        """Test that automatic clipping mode works end-to-end"""
        torch.manual_seed(42)
        model = SampleConvNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        criterion = nn.CrossEntropyLoss()
        data_loader = self._init_data()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.NOISE_MULTIPLIER,
            max_grad_norm=self.MAX_GRAD_NORM,
            poisson_sampling=False,
            clipping="automatic",
        )

        # Run one training step
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            break  # Just one batch

        # Verify optimizer is correct type
        self.assertIsInstance(optimizer, DPAutomaticClippingOptimizer)

    def test_automatic_per_layer_clipping_basic(self):
        """Test that automatic per-layer clipping mode works end-to-end"""
        torch.manual_seed(42)
        model = SampleConvNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        criterion = nn.CrossEntropyLoss()
        data_loader = self._init_data()

        # Get number of parameters for per-layer norms
        num_params = len(list(model.parameters()))
        max_grad_norm = [self.MAX_GRAD_NORM] * num_params

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.NOISE_MULTIPLIER,
            max_grad_norm=max_grad_norm,
            poisson_sampling=False,
            clipping="automatic_per_layer",
        )

        # Run one training step
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            break  # Just one batch

        # Verify optimizer is correct type
        self.assertIsInstance(optimizer, DPPerLayerAutomaticClippingOptimizer)

    def test_automatic_clipping_convergence(self):
        """Test that automatic clipping allows model to learn (loss decreases)"""
        torch.manual_seed(42)
        model = SampleConvNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR)
        criterion = nn.CrossEntropyLoss()
        data_loader = self._init_data()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.NOISE_MULTIPLIER,
            max_grad_norm=self.MAX_GRAD_NORM,
            poisson_sampling=False,
            clipping="automatic",
        )

        losses = []
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Check that loss decreased from first to last batch
        self.assertLess(losses[-1], losses[0])


if __name__ == "__main__":
    unittest.main()
