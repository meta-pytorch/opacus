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
import torch.nn as nn
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, TensorDataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 1)

    def forward(self, x):
        return self.fc(x)


class BatchMemoryManagerTest(unittest.TestCase):
    GSM_MODE = "hooks"

    def setUp(self) -> None:
        self.data_size = 256
        self.inps = torch.randn(self.data_size, 5)
        self.tgts = torch.randn(
            self.data_size,
        )

        self.dataset = TensorDataset(self.inps, self.tgts)

    def _init_training(self, batch_size=10, **data_loader_kwargs):
        model = Model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = DataLoader(
            self.dataset, batch_size=batch_size, **data_loader_kwargs
        )

        return model, optimizer, data_loader

    @given(
        num_workers=st.integers(0, 4),
        pin_memory=st.booleans(),
        batch_size=st.sampled_from([8, 16, 64]),
        max_physical_batch_size=st.sampled_from([4, 8]),
    )
    @settings(suppress_health_check=list(HealthCheck), deadline=60000)
    def test_basic(
        self,
        num_workers: int,
        pin_memory: bool,
        batch_size: int,
        max_physical_batch_size: int,
    ):
        batches_per_step = max(1, batch_size // max_physical_batch_size)
        model, optimizer, data_loader = self._init_training(
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size,
        )

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode=self.GSM_MODE,
        )
        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as new_data_loader:
            self.assertEqual(len(data_loader), len(data_loader.dataset) // batch_size)
            self.assertEqual(
                len(new_data_loader),
                len(data_loader.dataset) // max_physical_batch_size,
            )
            weights_before = torch.clone(model._module.fc.weight)
            for i, (x, y) in enumerate(new_data_loader):
                self.assertTrue(x.shape[0] <= max_physical_batch_size)

                out = model(x)
                loss = (y - out).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if (i + 1) % batches_per_step > 0:
                    self.assertTrue(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                else:
                    self.assertFalse(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                    weights_before = torch.clone(model._module.fc.weight)

    @pytest.mark.skip("Incompatible with the new empty batch handling")
    @given(
        num_workers=st.integers(0, 4),
        pin_memory=st.booleans(),
    )
    @settings(suppress_health_check=list(HealthCheck), deadline=60000)
    def test_empty_batch(
        self,
        num_workers: int,
        pin_memory: bool,
    ):
        batch_size = 2
        max_physical_batch_size = 10
        torch.manual_seed(30)

        model, optimizer, data_loader = self._init_training(
            num_workers=num_workers,
            pin_memory=pin_memory,
            batch_size=batch_size,
        )

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.0,
            max_grad_norm=1e5,
            poisson_sampling=True,
            grad_sample_mode=self.GSM_MODE,
        )
        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=optimizer,
        ) as new_data_loader:
            weights_before = torch.clone(model._module.fc.weight)
            for i, (x, y) in enumerate(new_data_loader):
                self.assertTrue(x.shape[0] <= max_physical_batch_size)

                out = model(x)
                loss = (y - out).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if len(x) == 0:
                    self.assertTrue(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                else:
                    self.assertFalse(
                        torch.allclose(model._module.fc.weight, weights_before)
                    )
                    weights_before = torch.clone(model._module.fc.weight)

    def test_equivalent_to_one_batch(self) -> None:
        torch.manual_seed(1337)
        model, optimizer, data_loader = self._init_training()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode=self.GSM_MODE,
        )

        with BatchMemoryManager(
            data_loader=data_loader, max_physical_batch_size=3, optimizer=optimizer
        ) as data_loader:
            for x, y in data_loader:
                out = model(x)
                loss = (y - out).mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        memory_manager_weights = model._module.fc.weight.detach()

        torch.manual_seed(1337)
        model, optimizer, data_loader = self._init_training()

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode=self.GSM_MODE,
        )

        for x, y in data_loader:
            out = model(x)
            loss = (y - out).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        vanilla_weights = model._module.fc.weight.detach()

        self.assertTrue(torch.allclose(memory_manager_weights, vanilla_weights))


class BatchMemoryManagerTestWithExpandedWeights(BatchMemoryManagerTest):
    GSM_MODE = "ew"

    def test_empty_batch(self) -> None:
        pass


class BatchMemoryManagerTestWithFunctorch(BatchMemoryManagerTest):
    GSM_MODE = "functorch"
