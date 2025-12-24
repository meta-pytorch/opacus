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
import torch.nn.functional as F
from opacus.grad_sample import GradSampleHooks, GradSampleModule
from opacus.grad_sample.linear import compute_linear_grad_sample
from opacus.grad_sample.utils import register_grad_sampler
from torch.testing import assert_close
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import mobilenet_v3_small


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # x of shape [B, 3, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class GradSampleHooksTest(unittest.TestCase):
    """Test GradSampleHooks - core logic approach without model wrapping."""

    CLS = GradSampleHooks

    def setUp(self):
        self.original_model = SampleConvNet()
        self.hooks_model = SampleConvNet()
        self.hooks_model.load_state_dict(self.original_model.state_dict(), strict=True)

        self.grad_sample_hooks = self.CLS(
            self.hooks_model, batch_first=True, loss_reduction="mean"
        )
        self.DATA_SIZE = 8
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(3, 28, 28),
            num_classes=10,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.DATA_SIZE)

    def tearDown(self):
        """Clean up hooks after each test."""
        if hasattr(self, "grad_sample_hooks"):
            self.grad_sample_hooks.cleanup()

    def test_outputs_unaltered(self):
        """
        Test that hooks won't alter any outputs.
        Model should behave identically with or without hooks.
        """
        x, _ = next(iter(self.dl))
        self.original_model = self.original_model.eval()
        self.hooks_model = self.hooks_model.eval()
        with torch.no_grad():
            normal_out = self.original_model(x)
            hooks_out = self.hooks_model(x)
        msg = (
            f"Hooks L2 norm = : {hooks_out.norm(2)}, ",
            f"Original L2 norm = : {normal_out.norm(2)}, ",
            f"MSE = {F.mse_loss(hooks_out, normal_out)}, ",
            f"L1 Loss = {F.l1_loss(hooks_out, normal_out)}",
        )
        assert_close(hooks_out, normal_out, atol=1e-7, rtol=1e-5, msg=msg)

    def test_model_not_wrapped(self):
        """Test that model is NOT wrapped - maintains original type."""
        # Model should be the original type, not wrapped
        assert isinstance(self.hooks_model, SampleConvNet)
        assert not isinstance(self.hooks_model, GradSampleModule)
        assert type(self.hooks_model).__name__ == "SampleConvNet"

    def test_remove_hooks(self):
        """
        Test that after calling .remove_hooks() no hooks are left
        """
        copy_of_original_model = SampleConvNet()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(),
            strict=True,
        )
        new_grad_sample_hooks = GradSampleHooks(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )
        new_grad_sample_hooks.remove_hooks()

        remaining_forward_hooks = {
            module: module._forward_hooks
            for module in copy_of_original_model.modules()
            if module._forward_hooks
        }
        assert (
            not remaining_forward_hooks
        ), f"Some forward hooks remain after .remove_hooks(): {remaining_forward_hooks}"

        remaining_backward_hooks = {
            module: module._backward_hooks
            for module in copy_of_original_model.modules()
            if module._backward_hooks
        }
        assert (
            not remaining_backward_hooks
        ), f"Some backward hooks remain after .remove_hooks(): {remaining_backward_hooks}"

        # Cleanup
        new_grad_sample_hooks.cleanup()

    def test_enable_hooks(self):
        """Test that hooks can be enabled."""
        self.grad_sample_hooks.enable_hooks()
        assert self.grad_sample_hooks.hooks_enabled

    def test_disable_hooks(self):
        """Test that hooks can be disabled."""
        self.grad_sample_hooks.disable_hooks()
        assert not self.grad_sample_hooks.hooks_enabled

    def test_standard_module_validation(self):
        """Test validation behavior for standard modules."""

        class SimpleLinear(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.p = nn.Parameter(torch.Tensor(in_f, out_f))

            def forward(self, x: torch.Tensor):
                return F.linear(x, self.p)

        # Should be handled by functorch
        try:
            hooks = GradSampleHooks(SimpleLinear(4, 2))
            # Check that functorch is used for this module
            self.assertTrue(hasattr(hooks._module, "ft_compute_sample_grad"))
            hooks.cleanup()
        except ImportError:
            print("Test could not be ran because functorch not available")

        # Should not raise exception if strict=False
        try:
            hooks = GradSampleHooks(SimpleLinear(4, 2), strict=False)
            hooks.cleanup()
        except ImportError:
            print("Test could not be ran because functorch not available")

        # Should not fail after relevant grad sampler has been registered
        register_grad_sampler(SimpleLinear)(compute_linear_grad_sample)
        hooks = GradSampleHooks(SimpleLinear(4, 2))
        hooks.cleanup()

    def test_custom_module_validation(self) -> None:
        """Test that unsupported modules raise appropriate errors."""
        from opacus.validators.errors import UnsupportedModuleError

        with self.assertRaises(UnsupportedModuleError):
            hooks = GradSampleHooks(mobilenet_v3_small())
            hooks.cleanup()

    def test_submodule_access(self) -> None:
        """Test that submodules can be accessed directly (no wrapping)."""
        # Direct access to submodules - no _module prefix needed
        _ = self.hooks_model.fc1
        _ = self.hooks_model.fc2

        with self.assertRaises(AttributeError):
            _ = self.hooks_model.fc3

    def test_state_dict(self) -> None:
        """Test that state_dict has no _module prefix (not wrapped)."""
        hooks_state_dict = self.hooks_model.state_dict()
        og_state_dict = self.original_model.state_dict()

        # Hooks approach: state dict keys should match exactly (no _module prefix)
        self.assertEqual(set(hooks_state_dict.keys()), set(og_state_dict.keys()))

        for key in og_state_dict.keys():
            # Keys should be identical, no _module prefix
            self.assertTrue(key in hooks_state_dict)
            assert_close(og_state_dict[key], hooks_state_dict[key])

    def test_load_state_dict(self) -> None:
        """Test that state_dict can be loaded without _module prefix."""
        hooks_state_dict = self.hooks_model.state_dict()
        new_model = SampleConvNet()
        new_hooks = GradSampleHooks(new_model, batch_first=False, loss_reduction="mean")

        # Should be able to load directly (no _module prefix)
        new_model.load_state_dict(hooks_state_dict)

        # Models should match
        for key in self.original_model.state_dict().keys():
            self.assertTrue(key in new_model.state_dict())
            assert_close(
                self.original_model.state_dict()[key], new_model.state_dict()[key]
            )

        new_hooks.cleanup()

    def test_grad_sample_computation(self):
        """Test that per-sample gradients are computed correctly."""
        x, _ = next(iter(self.dl))
        self.hooks_model.train()
        hooks_out = self.hooks_model(x)
        loss = self.criterion(hooks_out, torch.zeros_like(hooks_out))
        loss.backward()

        # Check that grad_sample was computed for all trainable parameters
        for name, param in self.hooks_model.named_parameters():
            if param.requires_grad:
                self.assertTrue(
                    hasattr(param, "grad_sample"),
                    f"Parameter {name} should have grad_sample",
                )
                self.assertIsNotNone(
                    param.grad_sample,
                    f"Parameter {name} grad_sample should not be None",
                )
                # grad_sample should have batch dimension
                self.assertEqual(
                    param.grad_sample.shape[0],
                    self.DATA_SIZE,
                    f"Parameter {name} grad_sample batch dimension mismatch",
                )

    def test_cleanup(self):
        """Test that cleanup removes all hooks and attributes."""
        x, _ = next(iter(self.dl))
        self.hooks_model.train()
        hooks_out = self.hooks_model(x)
        loss = self.criterion(hooks_out, torch.zeros_like(hooks_out))
        loss.backward()

        # Verify grad_sample exists
        for param in self.hooks_model.parameters():
            if param.requires_grad:
                self.assertTrue(hasattr(param, "grad_sample"))

        # Cleanup
        self.grad_sample_hooks.cleanup()

        # Verify attributes are removed
        for param in self.hooks_model.parameters():
            self.assertFalse(
                hasattr(param, "grad_sample"), "grad_sample should be removed"
            )
            self.assertFalse(
                hasattr(param, "_forward_counter"),
                "_forward_counter should be removed",
            )

        # Verify hooks are removed
        remaining_forward_hooks = {
            module: module._forward_hooks
            for module in self.hooks_model.modules()
            if module._forward_hooks
        }
        self.assertFalse(remaining_forward_hooks, "All forward hooks should be removed")

        remaining_backward_hooks = {
            module: module._backward_hooks
            for module in self.hooks_model.modules()
            if module._backward_hooks
        }
        self.assertFalse(
            remaining_backward_hooks, "All backward hooks should be removed"
        )

    def test_isinstance_preserved(self):
        """Test that isinstance checks work correctly with hooks (no wrapping)."""
        # Model should still be instance of original class
        self.assertIsInstance(self.hooks_model, SampleConvNet)
        self.assertIsInstance(self.hooks_model, nn.Module)

        # Should NOT be instance of GradSampleModule
        self.assertNotIsInstance(self.hooks_model, GradSampleModule)


if __name__ == "__main__":
    unittest.main()
