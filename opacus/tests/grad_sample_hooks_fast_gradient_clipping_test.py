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
from opacus.grad_sample import GradSampleHooksFastGradientClipping
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)


class SimpleModel(nn.Module):
    """Simple model for testing"""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GradSampleHooksFastGradientClippingTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.input_dim = 10
        self.hidden_dim = 20
        self.output_dim = 5
        self.max_grad_norm = 1.0
        self.loss_reduction = "mean"

    def test_hooks_creation(self):
        """Test that hooks can be created without wrapping model"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)
        original_type = type(model)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=True,
        )

        # Model type should be preserved
        self.assertEqual(type(model), original_type)
        self.assertIsInstance(model, SimpleModel)

        # Hooks should be installed
        self.assertTrue(len(hooks._module.autograd_grad_sample_hooks) > 0)
        self.assertTrue(hooks.hooks_enabled)

        # Clean up
        hooks.cleanup()

    def test_norm_sample_computation(self):
        """Test that norm samples are computed correctly"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,  # Use fast gradient clipping for testing
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(
            output, target, reduction=self.loss_reduction
        )
        loss.backward()

        # Check that norm samples are computed
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param._norm_sample)
                self.assertEqual(param._norm_sample.shape[0], self.batch_size)

        # Clean up
        hooks.cleanup()

    def test_clipping_coefficient(self):
        """Test that clipping coefficients are computed correctly"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(
            output, target, reduction=self.loss_reduction
        )
        loss.backward()

        # Get clipping coefficient
        coeff = hooks.get_clipping_coef()

        # Coefficients should be between 0 and 1
        self.assertTrue(torch.all(coeff >= 0))
        self.assertTrue(torch.all(coeff <= 1))
        self.assertEqual(coeff.shape[0], self.batch_size)

        # Clean up
        hooks.cleanup()

    def test_get_norm_sample(self):
        """Test that get_norm_sample returns correct per-example norms."""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(
            output, target, reduction=self.loss_reduction
        )
        loss.backward()

        # Get norms
        norms = hooks.get_norm_sample()

        # Check shape and values
        self.assertEqual(norms.shape[0], self.batch_size)
        self.assertTrue(torch.all(norms >= 0), "Norms should be non-negative")

        # Clean up
        hooks.cleanup()

    def test_grad_sample_computation(self):
        """Test that per-sample gradients are computed correctly even in ghost clipping mode."""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass
        model.train()
        output = model(x)
        loss = nn.functional.cross_entropy(
            output, target, reduction=self.loss_reduction
        )
        loss.backward()

        # Check that grad samples are computed then cleared
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNone(param.grad_sample)
                self.assertIsNotNone(param._norm_sample)

        # Clean up
        hooks.cleanup()

    def test_hooks_enable_disable(self):
        """Test that hooks can be enabled and disabled"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # Hooks should be enabled by default
        self.assertTrue(hooks.hooks_enabled)

        # Disable hooks
        hooks.disable_hooks()
        self.assertFalse(hooks.hooks_enabled)

        # Enable hooks
        hooks.enable_hooks()
        self.assertTrue(hooks.hooks_enabled)

        # Clean up
        hooks.cleanup()

    def test_cleanup(self):
        """Test that cleanup removes all hooks and attributes"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # Verify hooks and attributes exist
        self.assertTrue(len(hooks._module.autograd_grad_sample_hooks) > 0)
        for param in model.parameters():
            self.assertTrue(hasattr(param, "_forward_counter"))

        # Cleanup
        hooks.cleanup()

        # Verify hooks are removed
        self.assertFalse(hasattr(hooks._module, "autograd_grad_sample_hooks"))

        # Verify attributes are removed
        for param in model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))
            self.assertFalse(hasattr(param, "_forward_counter"))
            self.assertFalse(hasattr(param, "_norm_sample"))

    def test_hooks_vs_wrapped_equivalence(self):
        """Test that hooks produce same norms as wrapped module"""
        # Create two identical models
        model_hooks = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)
        model_wrapped = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Copy weights
        model_wrapped.load_state_dict(model_hooks.state_dict())

        # Create hooks and wrapped module
        hooks = GradSampleHooksFastGradientClipping(
            model_hooks,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,  # Use FGC for comparison
        )

        wrapped = GradSampleModuleFastGradientClipping(
            model_wrapped,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
            use_ghost_clipping=False,
        )

        # Create dummy input and target
        torch.manual_seed(42)
        x = torch.randn(self.batch_size, self.input_dim)
        target = torch.randint(0, self.output_dim, (self.batch_size,))

        # Forward and backward pass for hooks
        model_hooks.train()
        output_hooks = model_hooks(x.clone())
        loss_hooks = nn.functional.cross_entropy(
            output_hooks, target, reduction=self.loss_reduction
        )
        loss_hooks.backward()

        # Forward and backward pass for wrapped
        wrapped.train()
        output_wrapped = wrapped(x.clone())
        loss_wrapped = nn.functional.cross_entropy(
            output_wrapped, target, reduction=self.loss_reduction
        )
        loss_wrapped.backward()

        # Get norms
        norm_hooks = hooks.get_norm_sample()
        norm_wrapped = wrapped.get_norm_sample()

        # Norms should be very close
        self.assertTrue(torch.allclose(norm_hooks, norm_wrapped, rtol=1e-4, atol=1e-4))

        # Clean up
        hooks.cleanup()

    def test_isinstance_preserved(self):
        """Test that isinstance checks work after hooks attachment"""
        model = SimpleModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Before hooks
        self.assertIsInstance(model, SimpleModel)
        self.assertIsInstance(model, nn.Module)

        # Create hooks
        hooks = GradSampleHooksFastGradientClipping(
            model,
            batch_first=True,
            loss_reduction=self.loss_reduction,
            max_grad_norm=self.max_grad_norm,
        )

        # After hooks - isinstance should still work
        self.assertIsInstance(model, SimpleModel)
        self.assertIsInstance(model, nn.Module)

        # Clean up
        hooks.cleanup()


if __name__ == "__main__":
    unittest.main()
