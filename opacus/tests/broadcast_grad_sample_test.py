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

import copy
import unittest

import torch
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch import nn


class PositionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 3)
        self.position = nn.Embedding(4, 3)

    def forward(self, x, expand=False):
        ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        if expand:
            ids = ids.expand(x.shape[0], -1)
        return self.projection(x) + self.position(ids)


class BroadcastGradSampleTest(unittest.TestCase):
    def test_accumulated_and_empty_gradients(self):
        for sizes in ((0,), (2, 3)):
            with self.subTest(sizes=sizes):
                params = [nn.Parameter(torch.ones(2)) for _ in range(2)]
                optimizer = DPOptimizer(
                    torch.optim.SGD(params, lr=0.1),
                    noise_multiplier=0,
                    max_grad_norm=1,
                    expected_batch_size=5,
                    loss_reduction="sum",
                )
                for p in params:
                    p.grad_sample = [torch.ones(size, 2) for size in sizes]
                optimizer.step()
                # Each sample has four unit gradient coordinates across two parameters.
                expected = torch.ones(2) - 0.1 * sum(sizes) / (2 + 1e-6)
                for p in params:
                    torch.testing.assert_close(p, expected)

    def test_broadcast_fails_before_update(self):
        model = GradSampleModule(PositionModel(), loss_reduction="sum")
        optimizer = DPOptimizer(
            torch.optim.SGD(model.parameters(), lr=0.1),
            noise_multiplier=0,
            max_grad_norm=1,
            expected_batch_size=4,
            loss_reduction="sum",
        )
        before = [p.detach().clone() for p in model.parameters()]
        model(torch.randn(4, 4, 3)).square().sum().backward()
        self.assertEqual(model._module.position.weight.grad_sample.shape[0], 1)
        with self.assertRaisesRegex(
            ValueError, "inconsistent batch dimensions.*position_ids"
        ):
            optimizer.step()
        for p, original in zip(model.parameters(), before):
            torch.testing.assert_close(p, original)
            self.assertIsNone(p.summed_grad)

    def test_expanded_inputs_match_individual_gradients(self):
        for reduction in ("sum", "mean"):
            with self.subTest(reduction=reduction):
                torch.manual_seed(841)
                reference = PositionModel()
                model = GradSampleModule(
                    copy.deepcopy(reference), loss_reduction=reduction
                )
                x = torch.randn(4, 4, 3)
                individual = []
                for sample in x:
                    reference.zero_grad()
                    reference(sample.unsqueeze(0)).square().sum().backward()
                    individual.append([p.grad.clone() for p in reference.parameters()])
                loss = model(x, expand=True).square().sum()
                if reduction == "mean":
                    loss = loss / len(x)
                loss.backward()
                for i, p in enumerate(model.parameters()):
                    expected = torch.stack([g[i] for g in individual])
                    torch.testing.assert_close(p.grad_sample, expected)
                self.assertFalse(torch.allclose(individual[0][-1], individual[1][-1]))
                optimizer = DPOptimizer(
                    torch.optim.SGD(model.parameters(), lr=0.1),
                    noise_multiplier=0,
                    max_grad_norm=1,
                    expected_batch_size=len(x),
                    loss_reduction=reduction,
                )
                norms = torch.stack(
                    [
                        torch.cat([g.flatten() for g in sample]).norm()
                        for sample in individual
                    ]
                )
                factors = (1 / (norms + 1e-6)).clamp(max=1)
                before = [p.detach().clone() for p in model.parameters()]
                optimizer.step()
                for i, p in enumerate(model.parameters()):
                    clipped = sum(
                        f * sample[i] for f, sample in zip(factors, individual)
                    )
                    if reduction == "mean":
                        clipped = clipped / len(x)
                    torch.testing.assert_close(p, before[i] - 0.1 * clipped)

    def test_empty_mismatch_fails_before_accumulation(self):
        for sizes in ((0, 1), (1, 0)):
            with self.subTest(sizes=sizes):
                params = [nn.Parameter(torch.ones(2)) for _ in sizes]
                optimizer = DPOptimizer(
                    torch.optim.SGD(params, lr=0.1),
                    noise_multiplier=0,
                    max_grad_norm=1,
                    expected_batch_size=1,
                )
                for p, size in zip(params, sizes):
                    p.grad_sample = torch.zeros(size, 2)
                with self.assertRaisesRegex(
                    ValueError, "inconsistent batch dimensions"
                ):
                    optimizer.step()
                for p in params:
                    self.assertIsNone(p.summed_grad)
