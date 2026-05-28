#!/usr/bin/env python3
# Copyright (c) 2026 SlaClip authors.
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

import math
import unittest

import torch
from opacus import PrivacyEngine
from opacus.optimizers.optimizer import DPOptimizer
from torch.utils.data import DataLoader, TensorDataset

from research.slaclip.slaclipoptimizer import (
    SlaClipController,
    SlaClipDPOptimizer,
    paper_recommended_k,
)


def make_optimizer(
    optimizer_class=SlaClipDPOptimizer,
    *,
    noise_multiplier: float = 0.0,
    max_grad_norm: float = 2.0,
    expected_batch_size: int = 1,
    num_slots: int = 2,
    clipping_controller=None,
    generator=None,
    dtype=torch.float32,
):
    model = torch.nn.Linear(2, 1, bias=False, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    common_args = dict(
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        expected_batch_size=expected_batch_size,
        generator=generator,
    )
    if optimizer_class is SlaClipDPOptimizer:
        common_args.update(
            num_slots=num_slots,
            clipping_controller=clipping_controller,
        )
    return model, optimizer_class(optimizer, **common_args)


class SlaClipOptimizerResearchTest(unittest.TestCase):
    def test_paper_recommended_k_uses_equation_36(self):
        expected = {
            128: 8,
            256: 13,
            512: 21,
            1024: 34,
            2048: 54,
        }
        for batch_size, num_slots in expected.items():
            with self.subTest(batch_size=batch_size):
                self.assertEqual(paper_recommended_k(batch_size, 1.0), num_slots)

    def test_paper_recommended_k_accounts_for_noise_multiplier(self):
        batch_size = 2000
        noise_multiplier = 1.415787
        expected = math.floor(
            (batch_size / (2.0 * 2.576 * noise_multiplier)) ** (2.0 / 3.0)
        )

        self.assertEqual(paper_recommended_k(batch_size, noise_multiplier), expected)
        self.assertEqual(expected, 42)

    def test_optimizer_auto_selects_k_and_allows_explicit_override(self):
        _, automatic = make_optimizer(
            noise_multiplier=1.0,
            expected_batch_size=512,
            num_slots=None,
        )
        _, explicit = make_optimizer(
            noise_multiplier=1.0,
            expected_batch_size=512,
            num_slots=7,
        )

        self.assertEqual(automatic.K, 21)
        self.assertEqual(explicit.K, 7)

    def test_paper_recommended_k_rejects_invalid_inputs(self):
        invalid_inputs = (
            (0, 1.0),
            (128, 0.0),
            (128, float("inf")),
        )
        for batch_size, noise_multiplier in invalid_inputs:
            with self.subTest(
                batch_size=batch_size,
                noise_multiplier=noise_multiplier,
            ), self.assertRaises(ValueError):
                paper_recommended_k(batch_size, noise_multiplier)

    def test_paper_controller_equations_28_to_30(self):
        controller = SlaClipController(eta=0.5)
        slack_indicator = torch.tensor([0.2, 0.4])
        current_clip = 2.0
        adjusted_near_zero = 0.4 / current_clip
        target = 1.0 - (1.0 - adjusted_near_zero) / 2.0
        expected = current_clip * math.exp(0.5 * (target - 0.2))
        self.assertAlmostEqual(
            controller(current_clip, slack_indicator), expected, places=6
        )

    def test_controller_projects_near_zero_signal(self):
        controller = SlaClipController(
            eta=0.5,
            min_clipbound=0.1,
            max_clipbound=50.0,
        )
        current_clip = 2.0

        projected_low = controller(current_clip, torch.tensor([0.2, -100.0]))
        expected_low = current_clip * math.exp(0.5 * (0.5 - 0.2))
        self.assertAlmostEqual(projected_low, expected_low, places=6)

        projected_high = controller(current_clip, torch.tensor([0.2, 100.0]))
        expected_high = current_clip * math.exp(0.5 * (1.0 - 0.2))
        self.assertAlmostEqual(projected_high, expected_high, places=6)

    def test_controller_clamps_next_threshold(self):
        controller = SlaClipController(
            eta=1.0,
            min_clipbound=1.0,
            max_clipbound=3.0,
        )

        self.assertEqual(controller(2.0, torch.tensor([-100.0, 1.0])), 3.0)
        self.assertEqual(controller(2.0, torch.tensor([100.0, 1.0])), 1.0)

    def test_controller_allows_threshold_to_decrease(self):
        controller = SlaClipController(
            eta=0.5,
            min_clipbound=0.1,
            max_clipbound=50.0,
        )
        next_clip = controller(2.0, torch.tensor([0.9, 0.0]))

        self.assertGreater(next_clip, 0.0)
        self.assertLess(next_clip, 2.0)

    def test_controller_rejects_invalid_bounds_and_nonfinite_values(self):
        invalid_kwargs = (
            {"eta": float("inf")},
            {"min_clipbound": 0.0},
            {"min_clipbound": 1.0, "max_clipbound": 1.0},
            {"max_clipbound": float("inf")},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SlaClipController(**kwargs)

        controller = SlaClipController()
        with self.assertRaises(ValueError):
            controller(1.0, torch.tensor([float("nan"), 0.0]))

    def test_equations_7_and_8_preserve_extended_norm_bound(self):
        _, optimizer = make_optimizer(max_grad_norm=2.0, num_slots=4)
        norms = torch.tensor([0.0, 0.5, 1.25, 2.0, 3.0])
        slack = optimizer._encode_slack(norms, optimizer.current_clip)

        lambda_t = optimizer.current_clip / math.sqrt(optimizer.K)
        encoded_amount = slack.sum(dim=1)
        expected_amount = torch.clamp(
            optimizer.current_clip - norms, min=0.0
        ) * math.sqrt(optimizer.K)
        torch.testing.assert_close(encoded_amount, expected_amount)
        self.assertTrue(torch.all(slack >= 0))
        self.assertTrue(torch.all(slack <= lambda_t))

        clipped_norms = torch.clamp(norms, max=optimizer.current_clip)
        extended_norms = torch.sqrt(clipped_norms.square() + slack.square().sum(dim=1))
        self.assertTrue(torch.all(extended_norms <= optimizer.current_clip + 1e-6))

    def test_gradient_release_matches_native_dpoptimizer(self):
        native_model, native = make_optimizer(
            DPOptimizer,
            noise_multiplier=1.0,
            generator=torch.Generator().manual_seed(1234),
            dtype=torch.float64,
        )
        slaclip_model, slaclip = make_optimizer(
            noise_multiplier=1.0,
            generator=torch.Generator().manual_seed(1234),
            dtype=torch.float64,
        )
        grad_sample = torch.tensor([[[3.0, 4.0]], [[0.25, 0.5]]], dtype=torch.float64)
        next(native_model.parameters()).grad_sample = grad_sample.clone()
        next(slaclip_model.parameters()).grad_sample = grad_sample.clone()

        native.clip_and_accumulate()
        slaclip.clip_and_accumulate()
        torch.testing.assert_close(
            next(slaclip_model.parameters()).summed_grad,
            next(native_model.parameters()).summed_grad,
        )

        native.add_noise()
        slaclip.add_noise()
        torch.testing.assert_close(
            next(slaclip_model.parameters()).grad,
            next(native_model.parameters()).grad,
        )

    def test_indicator_only_mode_does_not_change_threshold(self):
        model, optimizer = make_optimizer(
            max_grad_norm=2.0,
            expected_batch_size=2,
            num_slots=4,
            clipping_controller=None,
        )
        parameter = next(model.parameters())
        parameter.grad_sample = torch.tensor(
            [[[0.0, 0.0]], [[1.0, 0.0]]], dtype=parameter.dtype
        )

        optimizer.clip_and_accumulate()
        optimizer.add_noise()

        self.assertEqual(optimizer.current_clip, 2.0)
        expected = torch.tensor([1.0, 1.0, 0.5, 0.5])
        torch.testing.assert_close(optimizer.slack_indicator, expected)

    def test_controller_consumes_private_indicator(self):
        model, optimizer = make_optimizer(
            max_grad_norm=2.0,
            expected_batch_size=2,
            num_slots=4,
            clipping_controller=SlaClipController(eta=0.5),
        )
        parameter = next(model.parameters())
        parameter.grad_sample = torch.tensor(
            [[[0.0, 0.0]], [[1.0, 0.0]]], dtype=parameter.dtype
        )

        optimizer.clip_and_accumulate()
        optimizer.add_noise()

        expected = SlaClipController(eta=0.5)(2.0, torch.tensor([1.0, 1.0, 0.5, 0.5]))
        self.assertAlmostEqual(optimizer.current_clip, expected, places=6)

    def test_empty_batch_releases_noisy_indicator(self):
        model, optimizer = make_optimizer(
            noise_multiplier=0.0,
            expected_batch_size=2,
            num_slots=2,
        )
        parameter = next(model.parameters())
        parameter.grad_sample = torch.empty(
            (0,) + tuple(parameter.shape), dtype=parameter.dtype
        )

        optimizer.clip_and_accumulate()
        optimizer.add_noise()

        torch.testing.assert_close(optimizer.slack_indicator, torch.zeros(2))

    def test_privacy_engine_records_one_joint_release(self):
        model = torch.nn.Linear(2, 1)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        data_loader = DataLoader(
            TensorDataset(torch.randn(8, 2), torch.randn(8, 1)), batch_size=4
        )
        privacy_engine = PrivacyEngine()
        model, private_optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=base_optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )
        optimizer = SlaClipDPOptimizer(
            private_optimizer.original_optimizer,
            noise_multiplier=private_optimizer.noise_multiplier,
            max_grad_norm=private_optimizer.max_grad_norm,
            expected_batch_size=private_optimizer.expected_batch_size,
            loss_reduction=private_optimizer.loss_reduction,
            generator=private_optimizer.generator,
            secure_mode=private_optimizer.secure_mode,
            num_slots=4,
            clipping_controller=SlaClipController(eta=0.5),
        )
        optimizer.attach_step_hook(private_optimizer.step_hook)

        inputs, targets = next(iter(data_loader))
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(inputs), targets).backward()
        optimizer.step()

        self.assertEqual(len(privacy_engine.accountant), 1)
        self.assertEqual(optimizer.slack_indicator.shape, (4,))


if __name__ == "__main__":
    unittest.main()
