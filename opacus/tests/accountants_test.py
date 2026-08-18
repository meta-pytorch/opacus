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

import hypothesis.strategies as st
from hypothesis import given, settings
from opacus.accountants import (
    GaussianAccountant,
    IAccountant,
    PRVAccountant,
    RDPAccountant,
    create_accountant,
    register_accountant,
    registry,
)
from opacus.accountants.utils import get_noise_multiplier


class AccountantRegistryTest(unittest.TestCase):

    class DummyAccountant(IAccountant):
        def __init__(self):
            pass

        def __len__(self):
            return 0

        def step(self, **kwargs):
            pass

        def get_epsilon(self, **kwargs):
            return 0.0

        def mechanism(cls) -> str:
            return "dummy"

    class Dummy2Accountant(DummyAccountant):
        pass

    def test_register_accountant(self) -> None:
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)
            self.assertIsInstance(
                create_accountant("dummy"), AccountantRegistryTest.DummyAccountant
            )
            self.assertEqual(create_accountant("dummy").mechanism(), "dummy")
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]

    def test_create_accountant_not_registered(self) -> None:
        with self.assertRaises(ValueError):
            create_accountant("not_registered")

    def test_register_existing_accountant(self):
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)

            with self.assertRaises(ValueError):
                register_accountant("rdp", AccountantRegistryTest.DummyAccountant)
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]

    def test_force_register_existing_accountant(self) -> None:
        try:
            register_accountant("dummy", AccountantRegistryTest.DummyAccountant)

            register_accountant(
                "dummy", AccountantRegistryTest.Dummy2Accountant, force=True
            )
            self.assertIsInstance(
                create_accountant("dummy"), AccountantRegistryTest.Dummy2Accountant
            )
        finally:
            if "dummy" in registry._ACCOUNTANTS:
                del registry._ACCOUNTANTS["dummy"]


class AccountingTest(unittest.TestCase):
    def test_rdp_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 7.32911117143)

    def test_gdp_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = GaussianAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertLess(6.59, epsilon)
        self.assertLess(epsilon, 6.6)

    def test_prv_accountant(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 // 0.04)

        accountant = PRVAccountant()

        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        epsilon = accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 6.777395712150674)

    def test_epsilon_non_negative_at_large_delta(self) -> None:
        # Fix for: https://github.com/meta-pytorch/opacus/issues/585
        # when delta >= delta(eps=0) eps must be clamped to 0 to avoid
        #   negatives(rdp, prv) or failing(gdp)
        for accountant, delta in (
            (PRVAccountant(), 0.08),
            (RDPAccountant(), 0.2),
            (GaussianAccountant(), 0.08),
        ):
            accountant.step(noise_multiplier=1.2, sample_rate=0.2)
            self.assertGreaterEqual(accountant.get_epsilon(delta=delta), 0.0)

    def test_len_counts_optimization_steps(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = 50

        for accountant in (RDPAccountant(), GaussianAccountant(), PRVAccountant()):
            for _ in range(steps):
                accountant.step(
                    noise_multiplier=noise_multiplier, sample_rate=sample_rate
                )
            self.assertEqual(len(accountant.history), 1)
            self.assertEqual(len(accountant), steps)

        for accountant in (RDPAccountant(), PRVAccountant()):
            for _ in range(steps):
                accountant.step(
                    noise_multiplier=noise_multiplier, sample_rate=sample_rate
                )
            accountant.step(
                noise_multiplier=noise_multiplier * 2, sample_rate=sample_rate
            )
            self.assertEqual(len(accountant.history), 2)
            self.assertEqual(len(accountant), steps + 1)

    def test_get_noise_multiplier_rdp_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="rdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.416, places=4)

    def test_get_noise_multiplier_rdp_steps(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        steps = 2000

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
        )

        self.assertAlmostEqual(noise_multiplier, 1.3562, places=4)

    def test_get_noise_multiplier_prv_epochs(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="prv",
        )

        self.assertAlmostEqual(noise_multiplier, 1.34765625, places=4)

    def test_get_noise_multiplier_prv_steps(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        steps = 2000

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=steps,
            accountant="prv",
        )

        self.assertAlmostEqual(noise_multiplier, 1.2915, places=4)

    @given(
        epsilon=st.floats(1.0, 10.0),
        epochs=st.integers(10, 100),
        sample_rate=st.sampled_from(
            [1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
        ),
        delta=st.sampled_from([1e-4, 1e-5, 1e-6]),
    )
    @settings(deadline=60000)
    def test_get_noise_multiplier_overshoot(self, epsilon, epochs, sample_rate, delta):
        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
        )

        accountant = create_accountant(mechanism="rdp")
        accountant.history = [
            (noise_multiplier, sample_rate, int(epochs / sample_rate))
        ]

        actual_epsilon = accountant.get_epsilon(delta=delta)
        self.assertLess(actual_epsilon, epsilon)

    def test_get_noise_multiplier_gdp(self) -> None:
        delta = 1e-5
        sample_rate = 0.04
        epsilon = 8
        epochs = 90

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=epochs,
            accountant="gdp",
        )

        self.assertAlmostEqual(noise_multiplier, 1.3232421875)

    def test_accountant_state_dict(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        dummy_dest = {"dummy_k": "dummy_v"}
        # history should be equal but not the same instance
        self.assertEqual(accountant.state_dict()["history"], accountant.history)
        self.assertFalse(accountant.state_dict()["history"] is accountant.history)
        # mechanism populated to supplied dict
        self.assertEqual(
            accountant.state_dict(dummy_dest)["mechanism"], accountant.mechanism()
        )
        # existing values in supplied dict unchanged
        self.assertEqual(
            accountant.state_dict(dummy_dest)["dummy_k"], dummy_dest["dummy_k"]
        )

    def test_accountant_load_state_dict(self) -> None:
        noise_multiplier = 1.5
        sample_rate = 0.04
        steps = int(90 / 0.04)

        accountant = RDPAccountant()
        for _ in range(steps - 1000):
            accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

        new_rdp_accountant = RDPAccountant()
        new_gdp_accountant = GaussianAccountant()
        # check corner cases
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({})
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({"1": 2})
        with self.assertRaises(ValueError):
            new_rdp_accountant.load_state_dict({"history": []})
        with self.assertRaises(ValueError):
            new_gdp_accountant.load_state_dict(accountant.state_dict())
        # check loading logic
        self.assertNotEqual(new_rdp_accountant.state_dict(), accountant.state_dict())
        new_rdp_accountant.load_state_dict(accountant.state_dict())
        self.assertEqual(new_rdp_accountant.state_dict(), accountant.state_dict())

        # ensure correct output after completion
        for _ in range(steps - 1000, steps):
            new_rdp_accountant.step(
                noise_multiplier=noise_multiplier, sample_rate=sample_rate
            )

        epsilon = new_rdp_accountant.get_epsilon(delta=1e-5)
        self.assertAlmostEqual(epsilon, 7.32911117143)
