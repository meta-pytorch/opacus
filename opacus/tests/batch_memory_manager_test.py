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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

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

    def _make_private(self, module, optimizer, data_loader):
        privacy_engine = PrivacyEngine()
        module, optimizer, _ = privacy_engine.make_private(
            module=module,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            grad_sample_mode=self.GSM_MODE,
        )
        return module, optimizer

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

    @unittest.skip("Incompatible with the new empty batch handling")
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

    def test_batch_memory_manager_multiple_dp_optimizers(self) -> None:
        batch_size = 16
        max_physical_batch_size = 4
        batches_per_step = max(1, batch_size // max_physical_batch_size)

        actor, actor_optimizer, data_loader = self._init_training(batch_size=batch_size)
        critic, critic_optimizer, _ = self._init_training(batch_size=batch_size)

        actor, actor_optimizer = self._make_private(actor, actor_optimizer, data_loader)
        critic, critic_optimizer = self._make_private(
            critic, critic_optimizer, data_loader
        )

        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=[actor_optimizer, critic_optimizer],
        ) as new_data_loader:
            actor_weights_before = torch.clone(actor._module.fc.weight)
            critic_weights_before = torch.clone(critic._module.fc.weight)
            for i, (x, y) in enumerate(new_data_loader):
                self.assertTrue(x.shape[0] <= max_physical_batch_size)

                actor_loss = (y - actor(x).squeeze(-1)).pow(2).mean()
                actor_loss.backward()
                actor_optimizer.step()
                actor_optimizer.zero_grad()

                critic_loss = (y - critic(x).squeeze(-1)).pow(2).mean()
                critic_loss.backward()
                critic_optimizer.step()
                critic_optimizer.zero_grad()

                if (i + 1) % batches_per_step > 0:
                    # Neither optimizer should step before the logical batch is over
                    self.assertTrue(
                        torch.allclose(actor._module.fc.weight, actor_weights_before)
                    )
                    self.assertTrue(
                        torch.allclose(critic._module.fc.weight, critic_weights_before)
                    )
                else:
                    self.assertFalse(
                        torch.allclose(actor._module.fc.weight, actor_weights_before)
                    )
                    self.assertFalse(
                        torch.allclose(critic._module.fc.weight, critic_weights_before)
                    )
                    actor_weights_before = torch.clone(actor._module.fc.weight)
                    critic_weights_before = torch.clone(critic._module.fc.weight)

    def test_batch_memory_manager_mixed_optimizers(self) -> None:
        # GAN-style setup: private discriminator + non-private generator
        batch_size = 16
        max_physical_batch_size = 4
        batches_per_step = max(1, batch_size // max_physical_batch_size)

        torch.manual_seed(1337)
        generator = Generator()
        discriminator, discriminator_optimizer, data_loader = self._init_training(
            batch_size=batch_size
        )
        generator_optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
        criterion = nn.BCEWithLogitsLoss()

        discriminator, discriminator_optimizer = self._make_private(
            discriminator, discriminator_optimizer, data_loader
        )

        with BatchMemoryManager(
            data_loader=data_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=[discriminator_optimizer, generator_optimizer],
        ) as new_data_loader:
            generator_weights_before = torch.clone(generator.fc.weight)
            discriminator_weights_before = torch.clone(discriminator._module.fc.weight)
            for i, (x, y) in enumerate(new_data_loader):
                self.assertTrue(x.shape[0] <= max_physical_batch_size)
                noise = torch.randn(x.shape[0], 5)

                # 1. Discriminator step: private, only steps at logical batch
                # boundaries, accumulates per sample gradients otherwise
                real_score = discriminator(x)
                fake_score = discriminator(generator(noise).detach())
                discriminator_loss = criterion(
                    real_score, torch.ones_like(real_score)
                ) + criterion(fake_score, torch.zeros_like(fake_score))
                discriminator_loss.backward()
                discriminator_optimizer.step()
                discriminator_optimizer.zero_grad()

                # 2. Generator step: standard optimizer without DP hooks, must
                # not be affected by signal_skip_step (no AttributeError) and
                # steps on every physical batch
                for p in discriminator.parameters():
                    p.requires_grad_(False)
                if hasattr(discriminator, "disable_hooks"):
                    # grad sample hooks expect a trainable module; pause them
                    # while gradients only flow through to the generator
                    discriminator.disable_hooks()
                fake_score = discriminator(generator(noise))
                generator_loss = criterion(fake_score, torch.ones_like(fake_score))
                generator_loss.backward()
                if hasattr(discriminator, "enable_hooks"):
                    discriminator.enable_hooks()
                for p in discriminator.parameters():
                    p.requires_grad_(True)
                generator_optimizer.step()
                generator_optimizer.zero_grad()

                # Standard optimizer has no skip mechanism: it steps every time
                self.assertFalse(
                    torch.allclose(generator.fc.weight, generator_weights_before)
                )
                generator_weights_before = torch.clone(generator.fc.weight)

                if (i + 1) % batches_per_step > 0:
                    self.assertTrue(
                        torch.allclose(
                            discriminator._module.fc.weight,
                            discriminator_weights_before,
                        )
                    )
                else:
                    self.assertFalse(
                        torch.allclose(
                            discriminator._module.fc.weight,
                            discriminator_weights_before,
                        )
                    )
                    discriminator_weights_before = torch.clone(
                        discriminator._module.fc.weight
                    )

    def test_batch_memory_manager_invalid_optimizer(self) -> None:
        _, _, data_loader = self._init_training()

        with self.assertRaises(TypeError):
            BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=2,
                optimizer=123,
            )
        with self.assertRaises(TypeError):
            BatchMemoryManager(
                data_loader=data_loader,
                max_physical_batch_size=2,
                optimizer=[
                    torch.optim.SGD(Model().parameters(), lr=0.1),
                    "not_an_optimizer",
                ],
            )


class BatchMemoryManagerTestWithExpandedWeights(BatchMemoryManagerTest):
    GSM_MODE = "ew"

    def test_empty_batch(self) -> None:
        pass


class BatchMemoryManagerTestWithFunctorch(BatchMemoryManagerTest):
    GSM_MODE = "functorch"
