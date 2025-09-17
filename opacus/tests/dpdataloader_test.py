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

import pytest
import torch
from opacus.data_loader import DPDataLoader
from torch.utils.data import DataLoader, TensorDataset


class DPDataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.data_size = 10
        self.dimension = 7
        self.num_classes = 11

    @pytest.mark.skip("Incompatible with the new empty batch handling")
    def test_collate_classes(self) -> None:
        x = torch.randn(self.data_size, self.dimension)
        y = torch.randint(low=0, high=self.num_classes, size=(self.data_size,))

        dataset = TensorDataset(x, y)
        data_loader = DPDataLoader(dataset, sample_rate=1e-5)

        x_b, y_b = next(iter(data_loader))
        self.assertEqual(x_b.size(0), 0)
        self.assertEqual(y_b.size(0), 0)

    @pytest.mark.skip("Incompatible with the new empty batch handling")
    def test_collate_tensor(self) -> None:
        x = torch.randn(self.data_size, self.dimension)

        dataset = TensorDataset(x)
        data_loader = DPDataLoader(dataset, sample_rate=1e-5)

        (s,) = next(iter(data_loader))

        self.assertEqual(s.size(0), 0)

    def test_drop_last_true(self) -> None:
        x = torch.randn(self.data_size, self.dimension)

        dataset = TensorDataset(x)
        data_loader = DataLoader(dataset, drop_last=True)
        _ = DPDataLoader.from_data_loader(data_loader)
