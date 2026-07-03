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

import operator
import unittest

import torch
import torch.fx as fx
import torch.nn as nn
import torchvision
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.validators.inplace import _is_inplace_node, fix_inplace_operations


def _clone(value):
    return value.clone() if torch.is_tensor(value) else value


def _inplace_node_count(module):
    """Count remaining in-place method/function nodes in a traced module."""
    if not isinstance(module, fx.GraphModule):
        return 0
    return sum(1 for node in module.graph.nodes if _is_inplace_node(node))


# Model classes that flow through ModuleValidator.fix() must be defined at module
# scope -- fix() clones via torch.save/load, which cannot pickle local classes.


class _OpModule(nn.Module):
    """Applies an in-place binary operator (e.g. ``operator.iadd``) to a fresh copy."""

    def __init__(self, inplace_op):
        super().__init__()
        self._inplace_op = inplace_op

    def forward(self, x, y):
        return self._inplace_op(x.clone(), y)


class _ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out += x  # in-place residual
        return self.act(out)


class _DynamicControlFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        out = self.fc(x)
        if out.sum() > 0:  # data-dependent control flow -> untraceable by fx
            out = out * 2
        out += x
        return out


class OperatorEquivalenceTest(unittest.TestCase):
    """Each augmented-assignment operator must give identical results before/after
    the rewrite (fx functionalizes these during tracing)."""

    def _cases(self):
        f = torch.randn(2, 3)
        g = torch.randn(2, 3)
        pos = torch.randn(2, 3).abs() + 1.0
        i = torch.randint(1, 8, (2, 3))
        j = torch.randint(1, 8, (2, 3))
        # (name, inplace_op, x, y, exact_compare)
        return [
            ("iadd", operator.iadd, f, g, False),
            ("isub", operator.isub, f, g, False),
            ("imul", operator.imul, f, g, False),
            ("itruediv", operator.itruediv, f, pos, False),
            ("ifloordiv", operator.ifloordiv, f, pos, False),
            ("imod", operator.imod, f.abs(), pos, False),
            ("ipow", operator.ipow, pos, 2.0, False),
            ("iand", operator.iand, i, j, True),
            ("ior", operator.ior, i, j, True),
            ("ixor", operator.ixor, i, j, True),
            ("ilshift", operator.ilshift, i, 1, True),
            ("irshift", operator.irshift, i, 1, True),
        ]

    def test_each_operator_is_equivalent(self):
        for name, inplace_op, x, y, exact in self._cases():
            with self.subTest(operator=name):
                module = _OpModule(inplace_op)
                eager = module(x.clone(), _clone(y))

                fixed = fix_inplace_operations(module)
                self.assertIsInstance(fixed, fx.GraphModule)
                # tracing made the augmented op out-of-place: nothing left in-place
                remaining = [
                    str(n.target) for n in fixed.graph.nodes if _is_inplace_node(n)
                ]
                self.assertEqual(remaining, [], f"{name}: remaining={remaining}")

                out = fixed(x.clone(), _clone(y))
                if exact:
                    self.assertTrue(torch.equal(eager, out))
                else:
                    self.assertTrue(torch.allclose(eager, out))


class MethodAndFunctionEquivalenceTest(unittest.TestCase):
    """In-place methods / functions / special cases preserve forward outputs and
    are rewritten out-of-place. These call fix_inplace_operations directly, so
    local model classes are fine (no clone/pickle)."""

    def _assert_equivalent(self, module, *inputs, exact=False):
        eager = module(*[_clone(i) for i in inputs])
        fixed = fix_inplace_operations(module)
        self.assertIsInstance(fixed, fx.GraphModule)
        self.assertEqual(_inplace_node_count(fixed), 0)
        out = fixed(*[_clone(i) for i in inputs])
        if exact:
            self.assertTrue(torch.equal(eager, out))
        else:
            self.assertTrue(torch.allclose(eager, out))
        return fixed

    def test_method_add(self):
        class M(nn.Module):
            def forward(self, x, y):
                return x.clone().add_(y)

        self._assert_equivalent(M(), torch.randn(2, 3), torch.randn(2, 3))

    def test_method_clamp(self):
        class M(nn.Module):
            def forward(self, x):
                return x.clone().clamp_(min=0.0)

        self._assert_equivalent(M(), torch.randn(2, 3))

    def test_method_masked_fill(self):
        class M(nn.Module):
            def forward(self, x, mask):
                return x.clone().masked_fill_(mask, 0.0)

        mask = torch.tensor([[True, False, True], [False, True, False]])
        self._assert_equivalent(M(), torch.randn(2, 3), mask)

    def test_functional_relu_(self):
        class M(nn.Module):
            def forward(self, x):
                return torch.relu_(x.clone())

        fixed = self._assert_equivalent(M(), torch.randn(2, 3))
        self.assertIn(torch.relu, [n.target for n in fixed.graph.nodes])

    def test_special_case_zero(self):
        class M(nn.Module):
            def forward(self, x):
                return x.clone().zero_()

        out = fix_inplace_operations(M())(torch.randn(2, 3))
        self.assertTrue(torch.equal(out, torch.zeros(2, 3)))

    def test_special_case_fill(self):
        class M(nn.Module):
            def forward(self, x):
                return x.clone().fill_(3.0)

        out = fix_inplace_operations(M())(torch.randn(2, 3))
        self.assertTrue(torch.equal(out, torch.full((2, 3), 3.0)))

    def test_special_case_copy(self):
        class M(nn.Module):
            def forward(self, x, src):
                return x.clone().copy_(src)

        src = torch.randn(2, 3)
        out = fix_inplace_operations(M())(torch.randn(2, 3), src.clone())
        self.assertTrue(torch.allclose(out, src))

    def test_special_case_copy_is_independent_of_src(self):
        # copy_ writes into x's own storage; the rewrite must not alias src.
        class M(nn.Module):
            def forward(self, x, src):
                return x.clone().copy_(src)

        src = torch.zeros(2, 3)
        out = fix_inplace_operations(M())(torch.randn(2, 3), src)
        src.add_(1.0)  # mutating src must not change a faithful copy_ result
        self.assertTrue(torch.equal(out, torch.zeros(2, 3)))

    def test_requires_grad_left_unchanged(self):
        # requires_grad_ strips to the requires_grad property (not callable), so it
        # must be left as-is rather than rewritten into a broken method call.
        class M(nn.Module):
            def forward(self, x):
                return x.clone().requires_grad_()

        with self.assertWarns(UserWarning):
            fixed = fix_inplace_operations(M())
        self.assertIn("requires_grad_", [str(n.target) for n in fixed.graph.nodes])
        self.assertEqual(fixed(torch.randn(2, 3)).shape, (2, 3))


class LeaveUnchangedTest(unittest.TestCase):
    """In-place ops with no safe out-of-place form are left as-is (and warned
    about), keeping their original semantics via tensor aliasing."""

    def test_statement_form_preserved_and_warns(self):
        class M(nn.Module):
            def forward(self, x, mask):
                x.masked_fill_(mask, 0.0)  # bare statement -> dropped by tracing
                return x.softmax(dim=-1)

        mask = torch.tensor([[True, False, True], [False, True, False]])
        x = torch.randn(2, 3)
        eager = M()(x.clone(), mask)
        with self.assertWarns(UserWarning):
            fixed = fix_inplace_operations(M())
        self.assertTrue(torch.allclose(eager, fixed(x.clone(), mask)))

    def test_out_kwarg_preserved_and_warns(self):
        class M(nn.Module):
            def forward(self, x, y):
                buf = torch.empty_like(x)
                torch.add(x, y, out=buf)
                return buf.relu()

        x, y = torch.randn(2, 3), torch.randn(2, 3)
        eager = M()(x.clone(), y.clone())
        with self.assertWarns(UserWarning):
            fixed = fix_inplace_operations(M())
        self.assertTrue(torch.allclose(eager, fixed(x.clone(), y.clone())))


class FixIntegrationTest(unittest.TestCase):
    """ModuleValidator.fix() behavior for in-place handling."""

    def test_inplace_flags_always_disabled(self):
        for remove_inplace_ops in (True, False):
            with self.subTest(remove_inplace_ops=remove_inplace_ops):
                model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(inplace=True))
                fixed = ModuleValidator.fix(
                    model, remove_inplace_ops=remove_inplace_ops
                )
                for m in fixed.modules():
                    if hasattr(m, "inplace"):
                        self.assertFalse(m.inplace)

    def test_residual_inplace_equivalence(self):
        model = _ResidualMLP().eval()
        x = torch.randn(4, 8)
        eager = model(x.clone())

        fixed = ModuleValidator.fix(model)
        self.assertIsInstance(fixed, fx.GraphModule)
        for m in fixed.modules():
            if hasattr(m, "inplace"):
                self.assertFalse(m.inplace)
        self.assertTrue(torch.allclose(eager, fixed.eval()(x.clone()), atol=1e-6))

    def test_fix_warns_when_returning_graph_module(self):
        model = _ResidualMLP()

        with self.assertWarnsRegex(
            UserWarning,
            "traced the model with FX.*GraphModule.*remove_inplace_ops = False",
        ):
            fixed = ModuleValidator.fix(model)

        self.assertIsInstance(fixed, fx.GraphModule)

    def test_untraceable_model_falls_back(self):
        model = _DynamicControlFlow()
        with self.assertLogs("opacus.validators.inplace", level="INFO"):
            fixed = ModuleValidator.fix(model, remove_inplace_ops=True)
        # tracing failed -> not a GraphModule, but still runnable
        self.assertNotIsInstance(fixed, fx.GraphModule)
        self.assertEqual(fixed(torch.randn(2, 4)).shape, (2, 4))


class ResNetEndToEndTest(unittest.TestCase):
    """The originally reported crash: ResNet under the default hooks mode."""

    def test_fixed_resnet_trains_under_hooks(self):
        model = torchvision.models.resnet18(num_classes=10)
        model = ModuleValidator.fix(model)
        self.assertTrue(ModuleValidator.is_valid(model))
        model.train()

        dataset = torch.utils.data.TensorDataset(
            torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
        )
        # Citrine C0: pin_memory=True for efficient CPU-to-GPU transfer
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, pin_memory=True
        )
        # Citrine C2: foreach=True for multi-tensor optimizer execution
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=True)

        privacy_engine = PrivacyEngine()
        model, optimizer, data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode="hooks",
        )

        criterion = nn.CrossEntropyLoss()
        data, target = next(iter(data_loader))
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()  # this is where the original crash happened
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
