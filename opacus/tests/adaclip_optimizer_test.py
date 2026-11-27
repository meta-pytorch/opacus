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

import unittest

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, TensorDataset


class AdaClipVirtualBatchTest(unittest.TestCase):
    def _run_logical_batch(self, inputs: torch.Tensor, max_physical_batch_size: int):
        torch.manual_seed(0)
        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        data_loader = DataLoader(
            TensorDataset(inputs, torch.zeros_like(inputs)), batch_size=len(inputs)
        )
        model, optimizer, data_loader = PrivacyEngine().make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            clipping="adaptive",
            target_unclipped_quantile=0.5,
            clipbound_learning_rate=0.2,
            max_clipbound=100.0,
            min_clipbound=0.01,
            unclipped_num_std=1e-9,
        )
        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as loader:
            for x, y in loader:
                optimizer.zero_grad()
                ((model(x) - y) ** 2).mean().backward()
                optimizer.step()
        return optimizer

    def test_counts_cover_whole_logical_batch_with_virtual_batches(self):
        # Per-sample grad norm is |2*w*x^2|: 100 is clipped, 0.1 is not.
        inputs = torch.tensor([10.0] * 4 + [0.1] * 4).unsqueeze(1)
        for max_physical_batch_size in (8, 4, 2):
            with self.subTest(max_physical_batch_size=max_physical_batch_size):
                optimizer = self._run_logical_batch(inputs, max_physical_batch_size)
                self.assertEqual(optimizer.sample_size, 8)
                self.assertAlmostEqual(float(optimizer.unclipped_num), 4.0, places=3)

    def test_counts_reset_after_a_completed_step(self):
        inputs = torch.tensor([0.1] * 8).unsqueeze(1)
        optimizer = self._run_logical_batch(inputs, 4)
        optimizer.zero_grad()
        self.assertEqual(optimizer.sample_size, 0)
        self.assertEqual(optimizer.unclipped_num, 0)


if __name__ == "__main__":
    unittest.main()
