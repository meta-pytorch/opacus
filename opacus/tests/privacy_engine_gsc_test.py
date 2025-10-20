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
"""
Comprehensive tests for PrivacyEngineGradSampleController.

This test suite mirrors the comprehensive tests in privacy_engine_test.py
but specifically for the controller-based implementation. Tests are reproduced
independently to validate that the controller-based approach provides identical
functionality to the standard wrapped approach.
"""

import math
import tempfile
import unittest
from typing import Optional, OrderedDict
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from opacus.optimizers.optimizer import _generate_noise
from opacus.privacy_engine_gsc import PrivacyEngineGradSampleController
from opacus.schedulers import StepGradClip, StepNoise
from opacus.utils.module_utils import are_state_dict_equal
from opacus.validators.errors import UnsupportedModuleError
from opacus.validators.module_validator import ModuleValidator
from opt_einsum import contract
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models


class SimpleNet(nn.Module):
    """Simple network for testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PrivacyEngineGradSampleControllerTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # Create a simple dataset
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=10)

    def test_initialization(self):
        """Test that PrivacyEngineGradSampleController can be initialized."""
        privacy_engine = PrivacyEngineGradSampleController()
        self.assertIsNotNone(privacy_engine)
        self.assertIsNotNone(privacy_engine.accountant)

    def test_make_private_returns_unwrapped_model(self):
        """Test that make_private returns the original model, not a wrapper."""
        privacy_engine = PrivacyEngineGradSampleController()

        original_model = self.model
        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Model should be the same object, not wrapped
        self.assertIs(model, original_model)
        self.assertIsInstance(model, SimpleNet)
        self.assertNotIsInstance(model, nn.Module.__class__)  # Not a wrapper

    def test_hooks_are_attached(self):
        """Test that hooks are properly attached to the model."""
        privacy_engine = PrivacyEngineGradSampleController()

        controller, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            return_controller=True,
        )

        # Check that hook controller was created
        self.assertIsNotNone(controller)
        self.assertTrue(len(controller.autograd_grad_sample_hooks) > 0)
        controller.cleanup()

    def test_grad_sample_computation(self):
        """Test that per-sample gradients are computed correctly."""
        privacy_engine = PrivacyEngineGradSampleController()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,  # Disable for simpler test
        )

        # Get a batch
        data, target = next(iter(dataloader))

        # Forward pass
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)

        # Backward pass
        loss.backward()

        # Check that grad_sample was computed
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad_sample)
                # grad_sample should have batch dimension
                self.assertEqual(param.grad_sample.shape[0], data.shape[0])

    def test_optimizer_step(self):
        """Test that optimizer step works with controller-based approach."""
        privacy_engine = PrivacyEngineGradSampleController()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
        )

        # Get a batch
        data, target = next(iter(dataloader))

        # Training step
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Store original parameters
        original_params = [p.clone() for p in model.parameters()]

        # Optimizer step should work
        optimizer.step()
        optimizer.zero_grad()

        # Parameters should have changed
        for original_param, current_param in zip(original_params, model.parameters()):
            self.assertFalse(torch.allclose(original_param, current_param))

    def test_cleanup(self):
        """Test that cleanup removes hooks and attributes."""
        privacy_engine = PrivacyEngineGradSampleController()

        controller, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            return_controller=True,
        )

        # Cleanup
        controller.cleanup()

        # Parameters should not have grad_sample attribute
        for param in self.model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))

    def test_state_dict_unchanged(self):
        """Test that state_dict remains unchanged (no wrapper prefix)."""
        privacy_engine = PrivacyEngineGradSampleController()

        # Get state dict before making private
        state_dict_before = self.model.state_dict()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Get state dict after making private
        state_dict_after = model.state_dict()

        # Keys should be identical (no _module prefix)
        self.assertEqual(set(state_dict_before.keys()), set(state_dict_after.keys()))

        # Values should be identical
        for key in state_dict_before.keys():
            self.assertTrue(
                torch.allclose(state_dict_before[key], state_dict_after[key])
            )

    def test_model_attribute_access(self):
        """Test that model attributes can be accessed directly."""
        privacy_engine = PrivacyEngineGradSampleController()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Should be able to access model attributes directly
        self.assertIsInstance(model.fc1, nn.Linear)
        self.assertIsInstance(model.fc2, nn.Linear)

        # Should be able to check module type
        self.assertIsInstance(model, SimpleNet)

    def test_checkpoint_save_load(self):
        """Test that checkpoints can be saved and loaded."""
        import tempfile

        privacy_engine = PrivacyEngineGradSampleController()

        model, optimizer, dataloader = privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Train for one step
        data, target = next(iter(dataloader))
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False) as f:
            checkpoint_path = f.name
            privacy_engine.save_checkpoint(
                path=checkpoint_path,
                module=model,
                optimizer=optimizer,
            )

        # Create new engine and model
        new_privacy_engine = PrivacyEngineGradSampleController()
        new_model = SimpleNet()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

        new_model, new_optimizer, _ = new_privacy_engine.make_private(
            module=new_model,
            optimizer=new_optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )

        # Load checkpoint
        new_privacy_engine.load_checkpoint(
            path=checkpoint_path,
            module=new_model,
            optimizer=new_optimizer,
        )

        # Parameters should match
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        # Clean up
        import os

        os.unlink(checkpoint_path)

    def test_manual_cleanup(self):
        """Test that manual cleanup properly cleans up."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        privacy_engine = PrivacyEngineGradSampleController()
        controller, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            return_controller=True,
        )

        # Hook controller should exist
        self.assertIsNotNone(controller)

        # Train one step
        data, target = next(iter(dataloader))
        model.train()
        output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()

        # Cleanup
        controller.cleanup()

        # Parameters should not have grad_sample
        for param in model.parameters():
            self.assertFalse(hasattr(param, "grad_sample"))

    def test_cleanup_with_exception(self):
        """Test that cleanup can happen even with exceptions."""
        model = SimpleNet()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        privacy_engine = PrivacyEngineGradSampleController()
        controller, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            return_controller=True,
        )

        # Hook controller should exist
        self.assertIsNotNone(controller)

        # Even if an exception occurs, cleanup should still be possible
        try:
            raise ValueError("Test exception")
        except ValueError:
            controller.cleanup()  # Manual cleanup still works

    def test_ddp_model_handling(self):
        """Test that DDP-wrapped models are properly handled."""
        model = SimpleNet()

        # Note: We use a mock DDP that doesn't inherit from actual DDP
        # In this case, the model is treated as a regular module (not unwrapped)
        # Real DDP instances would be unwrapped by GradSampleController
        class MockDDP(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, x):
                return self.module(x)

            def parameters(self, recurse=True):
                return self.module.parameters(recurse=recurse)

        ddp_model = MockDDP(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)

        privacy_engine = PrivacyEngineGradSampleController()

        # Make private with DDP model
        controller, optimizer, dataloader = privacy_engine.make_private(
            module=ddp_model,
            optimizer=optimizer,
            data_loader=self.dataloader,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            poisson_sampling=False,
            return_controller=True,
        )

        # Hook controller should exist
        # Since MockDDP is not a real DDP, target_module will be the MockDDP itself
        self.assertIsNotNone(controller)
        # For real DDP instances, this would be unwrapped to model
        # But MockDDP is treated as a regular module
        self.assertIs(controller.module, ddp_model)

        # Should be able to train
        data, target = next(iter(dataloader))
        ddp_model.train()
        output = ddp_model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()

        # Grad samples should be on underlying module's parameters
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad_sample)

        controller.cleanup()


def get_grad_sample_aggregated(tensor: torch.Tensor, loss_type: str = "mean"):
    """Helper to aggregate grad_sample for comparison."""
    if tensor.grad_sample is None:
        raise ValueError(
            f"The input tensor {tensor} has grad computed, but missing grad_sample."
        )

    if loss_type not in ("sum", "mean"):
        raise ValueError(f"loss_type = {loss_type}. Only 'sum' and 'mean' supported")

    grad_sample_aggregated = contract("i...->...", tensor.grad_sample)
    if loss_type == "mean":
        b_sz = tensor.grad_sample.shape[0]
        grad_sample_aggregated /= b_sz

    return grad_sample_aggregated


class BasePrivacyEngineGradSampleControllerTest:
    """
    Base class for controller-based privacy engine tests.

    Similar to BasePrivacyEngineTest but specifically for PrivacyEngineGradSampleController.
    Subclasses should implement _init_model() and _init_data().
    """

    def setUp(self):
        self.DATA_SIZE = 512
        self.BATCH_SIZE = 64
        self.LR = 0.5
        self.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        self.criterion = nn.CrossEntropyLoss()
        self.BATCH_FIRST = True
        self.GRAD_SAMPLE_MODE = "hooks"

        torch.manual_seed(42)

    def tearDown(self):
        """Clean up controller-based engine."""
        if hasattr(self, "controller") and self.controller is not None:
            self.controller.cleanup()

    def _init_vanilla_training(
        self,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
        opt_exclude_frozen=False,
    ):
        model = self._init_model()
        optimizer = torch.optim.SGD(
            (
                model.parameters()
                if not opt_exclude_frozen
                else [p for p in model.parameters() if p.requires_grad]
            ),
            lr=self.LR,
            momentum=0,
        )
        if state_dict:
            model.load_state_dict(state_dict)
        dl = self._init_data()
        return model, optimizer, dl

    def _init_private_training(
        self,
        state_dict: Optional[OrderedDict[str, torch.Tensor]] = None,
        secure_mode: bool = False,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        poisson_sampling: bool = True,
        clipping: str = "flat",
        grad_sample_mode="hooks",
        opt_exclude_frozen=False,
    ):
        model = self._init_model()
        model = PrivacyEngineGradSampleController.get_compatible_module(model)
        optimizer = torch.optim.SGD(
            (
                model.parameters()
                if not opt_exclude_frozen
                else [p for p in model.parameters() if p.requires_grad]
            ),
            lr=self.LR,
            momentum=0,
        )

        if state_dict:
            model.load_state_dict(state_dict)

        dl = self._init_data()

        if clipping == "per_layer":
            num_layers = len([p for p in model.parameters() if p.requires_grad])
            max_grad_norm = [max_grad_norm] * num_layers

        self.privacy_engine = PrivacyEngineGradSampleController(secure_mode=secure_mode)
        self.controller, optimizer, poisson_dl = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=self.BATCH_FIRST,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            return_controller=True,
        )

        return model, optimizer, poisson_dl, self.privacy_engine

    def _train_steps(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        dl: DataLoader,
        max_steps: Optional[int] = None,
    ):
        steps = 0
        epochs = 1 if max_steps is None else math.ceil(max_steps / len(dl))

        for _ in range(epochs):
            for x, y in dl:
                if optimizer:
                    optimizer.zero_grad()
                logits = model(x)
                loss = self.criterion(logits, y)
                loss.backward()
                if optimizer:
                    optimizer.step()

                steps += 1
                if max_steps and steps >= max_steps:
                    break

    def test_basic(self) -> None:
        """Basic training test."""
        for opt_exclude_frozen in [True, False]:
            with self.subTest(opt_exclude_frozen=opt_exclude_frozen):
                model, optimizer, dl, _ = self._init_private_training(
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                    poisson_sampling=True,
                    grad_sample_mode=self.GRAD_SAMPLE_MODE,
                    opt_exclude_frozen=opt_exclude_frozen,
                )

                self._train_steps(model, optimizer, dl, max_steps=1)

    def test_basic_noise0(self) -> None:
        """Test with zero noise."""
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=0,
            max_grad_norm=1.0,
            poisson_sampling=True,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

        self._train_steps(model, optimizer, dl, max_steps=1)

    def test_sample_grad_aggregation(self) -> None:
        """Test that grad_sample aggregation matches regular grad."""
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=0,
            max_grad_norm=999,
            poisson_sampling=False,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

        model.train()
        optimizer.zero_grad()
        x, y = next(iter(dl))
        logits = model(x)
        loss = self.criterion(logits, y)
        loss.backward()

        # Compare grad_sample aggregation with regular grad
        for p in model.parameters():
            if p.requires_grad:
                grad_sample_aggregated = get_grad_sample_aggregated(p, "mean")
                self.assertTrue(
                    torch.allclose(
                        p.grad, grad_sample_aggregated, atol=10e-5, rtol=10e-3
                    ),
                    f"Gradient mismatch. Expected {p.grad}, got {grad_sample_aggregated}",
                )

    def test_controller_cleanup(self) -> None:
        """Test manual hook controller cleanup."""
        model = self._init_model()
        model = PrivacyEngineGradSampleController.get_compatible_module(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.LR, momentum=0)
        dl = self._init_data()

        privacy_engine = PrivacyEngineGradSampleController()
        controller, optimizer, dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            return_controller=True,
        )

        self._train_steps(model, optimizer, dl, max_steps=1)

        # Should have hook controller
        self.assertIsNotNone(controller)

        # Manual cleanup
        controller.cleanup()

    def test_state_dict_no_wrapper_prefix(self) -> None:
        """Test that state dict has no _module prefix (not wrapped)."""
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

        state_dict = model.state_dict()
        for key in state_dict.keys():
            self.assertFalse(
                key.startswith("_module."),
                f"State dict key {key} should not have _module prefix",
            )

    def test_checkpoint_save_load(self) -> None:
        """Test checkpoint save and load."""
        model, optimizer, dl, privacy_engine = self._init_private_training(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

        # Train one step
        self._train_steps(model, optimizer, dl, max_steps=1)

        # Save checkpoint
        with tempfile.NamedTemporaryFile(delete=False) as f:
            checkpoint_path = f.name
            privacy_engine.save_checkpoint(
                path=checkpoint_path,
                module=model,
                optimizer=optimizer,
            )

        # Load into new model
        model2, optimizer2, dl2, privacy_engine2 = self._init_private_training(
            noise_multiplier=1.0,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

        privacy_engine2.load_checkpoint(
            path=checkpoint_path,
            module=model2,
            optimizer=optimizer2,
        )

        # State should match
        self.assertTrue(are_state_dict_equal(model.state_dict(), model2.state_dict()))

        # Clean up
        import os

        os.unlink(checkpoint_path)

    def test_flat_clipping(self) -> None:
        """Test flat gradient clipping."""
        orig_batch_size = self.BATCH_SIZE
        self.BATCH_SIZE = 1
        max_grad_norm = 0.5

        torch.manual_seed(1337)
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm,
            clipping="flat",
            poisson_sampling=False,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        self._train_steps(model, optimizer, dl, max_steps=1)
        clipped_grads = torch.cat(
            [p.summed_grad.reshape(-1) for p in model.parameters() if p.requires_grad]
        )

        torch.manual_seed(1337)
        model, optimizer, dl = self._init_vanilla_training()
        self._train_steps(model, optimizer, dl, max_steps=1)
        non_clipped_grads = torch.cat(
            [p.grad.reshape(-1) for p in model.parameters() if p.requires_grad]
        )

        self.assertAlmostEqual(clipped_grads.norm().item(), max_grad_norm, places=3)
        self.assertGreater(non_clipped_grads.norm(), clipped_grads.norm())

        # Restore batch size
        self.BATCH_SIZE = orig_batch_size

    def test_per_layer_clipping(self) -> None:
        """Test per-layer gradient clipping."""
        orig_batch_size = self.BATCH_SIZE
        self.BATCH_SIZE = 1
        max_grad_norm_per_layer = 1.0

        torch.manual_seed(1337)
        p_model, p_optimizer, p_dl, _ = self._init_private_training(
            noise_multiplier=0.0,
            max_grad_norm=max_grad_norm_per_layer,
            clipping="per_layer",
            poisson_sampling=False,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        p_optimizer.signal_skip_step()
        self._train_steps(p_model, p_optimizer, p_dl, max_steps=1)

        torch.manual_seed(1337)
        v_model, v_optimizer, v_dl = self._init_vanilla_training()
        self._train_steps(v_model, v_optimizer, v_dl, max_steps=1)

        for p_p, v_p in zip(p_model.parameters(), v_model.parameters()):
            if not p_p.requires_grad:
                continue

            non_clipped_norm = v_p.grad.norm().item()
            clipped_norm = p_p.summed_grad.norm().item()

            self.assertAlmostEqual(
                min(non_clipped_norm, max_grad_norm_per_layer), clipped_norm, places=3
            )

        # Restore batch size
        self.BATCH_SIZE = orig_batch_size

    def test_noise_changes_every_time(self) -> None:
        """Test that noise is different across runs."""
        model, optimizer, dl, _ = self._init_private_training(
            poisson_sampling=False, grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(model, optimizer, dl, max_steps=1)
        first_run_params = [p.clone() for p in model.parameters() if p.requires_grad]

        model, optimizer, dl, _ = self._init_private_training(
            poisson_sampling=False, grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(model, optimizer, dl, max_steps=1)
        second_run_params = [p for p in model.parameters() if p.requires_grad]

        for p0, p1 in zip(first_run_params, second_run_params):
            self.assertFalse(torch.allclose(p0, p1))

    def test_deterministic_run(self) -> None:
        """Test deterministic behavior with same seed."""
        torch.manual_seed(0)
        m1, opt1, dl1, _ = self._init_private_training(
            grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(m1, opt1, dl1, max_steps=2)
        params1 = [p.clone() for p in m1.parameters() if p.requires_grad]

        torch.manual_seed(0)
        m2, opt2, dl2, _ = self._init_private_training(
            grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(m2, opt2, dl2, max_steps=2)
        params2 = [p for p in m2.parameters() if p.requires_grad]

        for p1, p2 in zip(params1, params2):
            self.assertTrue(
                torch.allclose(p1, p2),
                "Model parameters after deterministic run must match",
            )

    def _compare_to_vanilla(
        self,
        do_noise,
        do_clip,
        expected_match,
        grad_sample_mode,
        use_closure=False,
        max_steps=1,
    ):
        """Compare private training to vanilla training."""
        torch.manual_seed(0)
        v_model, v_optimizer, v_dl = self._init_vanilla_training()
        if not use_closure:
            self._train_steps(v_model, v_optimizer, v_dl, max_steps=max_steps)
        else:
            self._train_steps_with_closure(
                v_model, v_optimizer, v_dl, max_steps=max_steps
            )
        vanilla_params = [
            (name, p) for name, p in v_model.named_parameters() if p.requires_grad
        ]

        torch.manual_seed(0)
        p_model, p_optimizer, p_dl, _ = self._init_private_training(
            poisson_sampling=False,
            noise_multiplier=1.0 if do_noise else 0.0,
            max_grad_norm=0.1 if do_clip else 1e20,
            grad_sample_mode=grad_sample_mode,
        )
        if not use_closure:
            self._train_steps(p_model, p_optimizer, p_dl, max_steps=max_steps)
        else:
            self._train_steps_with_closure(
                p_model, p_optimizer, p_dl, max_steps=max_steps
            )
        private_params = [p for p in p_model.parameters() if p.requires_grad]

        for (name, vp), pp in zip(vanilla_params, private_params):
            if vp.grad.norm() < 1e-4:
                # vanilla gradient is nearly zero: will match even with clipping
                continue

            atol = 1e-7 if max_steps == 1 else 1e-4
            self.assertEqual(
                torch.allclose(vp, pp, atol=atol, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla weight match ({name})."
                f"Should be: {expected_match}",
            )
            self.assertEqual(
                torch.allclose(vp.grad, pp.grad, atol=atol, rtol=1e-3),
                expected_match,
                f"Unexpected private/vanilla gradient match ({name})."
                f"Should be: {expected_match}",
            )

    def _train_steps_with_closure(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dl: DataLoader,
        max_steps: Optional[int] = None,
    ):
        """Train with closure-based optimizer step."""
        steps = 0
        epochs = 1 if max_steps is None else math.ceil(max_steps / len(dl))

        for _ in range(epochs):
            for x, y in dl:

                def closure():
                    optimizer.zero_grad()
                    logits = model(x)
                    loss = self.criterion(logits, y)
                    loss.backward()
                    return loss

                optimizer.step(closure)

                steps += 1
                if max_steps and steps >= max_steps:
                    break

    @given(
        do_clip=st.booleans(),
        do_noise=st.booleans(),
        use_closure=st.booleans(),
        max_steps=st.sampled_from([1, 3]),
    )
    @settings(suppress_health_check=list(HealthCheck), deadline=None)
    def test_compare_to_vanilla(
        self,
        do_clip: bool,
        do_noise: bool,
        use_closure: bool,
        max_steps: int,
    ):
        """Compare gradients and weights with vanilla model."""
        self._compare_to_vanilla(
            do_noise=do_noise,
            do_clip=do_clip,
            expected_match=not (do_noise or do_clip),
            use_closure=use_closure,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            max_steps=max_steps,
        )

    def test_get_compatible_module_inaction(self) -> None:
        """Test that get_compatible_module creates a copy."""
        needs_no_replacement_module = nn.Linear(1, 2)
        fixed_module = PrivacyEngineGradSampleController.get_compatible_module(
            needs_no_replacement_module
        )
        self.assertFalse(fixed_module is needs_no_replacement_module)
        self.assertTrue(
            are_state_dict_equal(
                needs_no_replacement_module.state_dict(), fixed_module.state_dict()
            )
        )

    def test_model_validator(self) -> None:
        """Test that privacy engine raises errors for unsupported modules."""
        resnet = models.resnet18()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        dl = self._init_data()
        privacy_engine = PrivacyEngineGradSampleController()
        with self.assertRaises(UnsupportedModuleError):
            _, _, _ = privacy_engine.make_private(
                module=resnet,
                optimizer=optimizer,
                data_loader=dl,
                noise_multiplier=1.3,
                max_grad_norm=1,
                grad_sample_mode=self.GRAD_SAMPLE_MODE,
            )

    def test_model_validator_after_fix(self) -> None:
        """Test that privacy engine works after fixing unsupported modules."""
        resnet = PrivacyEngineGradSampleController.get_compatible_module(
            models.resnet18()
        )
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        dl = self._init_data()
        privacy_engine = PrivacyEngineGradSampleController()
        controller, _, _ = privacy_engine.make_private(
            module=resnet,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.3,
            max_grad_norm=1,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            return_controller=True,
        )
        self.assertTrue(True)
        controller.cleanup()

    def test_make_private_with_epsilon(self) -> None:
        """Test make_private_with_epsilon method."""
        model, optimizer, dl = self._init_vanilla_training()
        target_eps = 2.0
        target_delta = 1e-5
        epochs = 2
        total_steps = epochs * len(dl)

        privacy_engine = PrivacyEngineGradSampleController()
        controller, optimizer, poisson_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            target_epsilon=target_eps,
            target_delta=1e-5,
            epochs=epochs,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            return_controller=True,
        )
        self._train_steps(model, optimizer, poisson_dl, max_steps=total_steps)
        self.assertAlmostEqual(
            target_eps, privacy_engine.get_epsilon(target_delta), places=2
        )
        controller.cleanup()

    def test_parameters_match(self) -> None:
        """Test that optimizer and model parameters must be same objects."""
        dl = self._init_data()

        m1 = self._init_model()
        m2 = self._init_model()
        m2.load_state_dict(m1.state_dict())
        # optimizer is initialized with m2 parameters
        opt = torch.optim.SGD(m2.parameters(), lr=0.1)

        # the values are identical
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        privacy_engine = PrivacyEngineGradSampleController()
        # but model parameters and optimizer parameters must be the same object,
        # not just same values
        with self.assertRaises(ValueError):
            privacy_engine.make_private(
                module=m1,
                optimizer=opt,
                data_loader=dl,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
                grad_sample_mode=self.GRAD_SAMPLE_MODE,
            )

    @given(
        has_noise_scheduler=st.booleans(),
        has_grad_clip_scheduler=st.booleans(),
    )
    @settings(suppress_health_check=list(HealthCheck), deadline=None)
    def test_checkpoints_with_schedulers(
        self, has_noise_scheduler: bool, has_grad_clip_scheduler: bool
    ):
        """Test checkpoint saving/loading with schedulers."""
        # 1. Disable poisson sampling to avoid randomness
        # 2. Use noise_multiplier=0.0 to avoid randomness in torch.normal()
        torch.manual_seed(1)
        m1, opt1, dl1, pe1 = self._init_private_training(
            noise_multiplier=0.0,
            poisson_sampling=False,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        noise_scheduler1 = (
            StepNoise(optimizer=opt1, step_size=1, gamma=1.0)
            if has_noise_scheduler
            else None
        )
        grad_clip_scheduler1 = (
            StepGradClip(optimizer=opt1, step_size=1, gamma=1.0)
            if has_grad_clip_scheduler
            else None
        )

        # create a different set of components: set 2
        torch.manual_seed(2)
        m2, opt2, _, pe2 = self._init_private_training(
            noise_multiplier=2.0,
            poisson_sampling=False,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        noise_scheduler2 = (
            StepNoise(optimizer=opt2, step_size=1, gamma=2.0)
            if has_noise_scheduler
            else None
        )
        grad_clip_scheduler2 = (
            StepGradClip(optimizer=opt2, step_size=1, gamma=2.0)
            if has_grad_clip_scheduler
            else None
        )

        # check that two sets of components are different
        self.assertFalse(are_state_dict_equal(m1.state_dict(), m2.state_dict()))
        if has_noise_scheduler:
            self.assertNotEqual(
                noise_scheduler1.state_dict(), noise_scheduler2.state_dict()
            )

        if has_grad_clip_scheduler:
            self.assertNotEqual(
                grad_clip_scheduler1.state_dict(), grad_clip_scheduler2.state_dict()
            )

        self.assertNotEqual(opt1.noise_multiplier, opt2.noise_multiplier)

        # train set 1 for a few steps
        self._train_steps(m1, opt1, dl1, max_steps=2)
        if has_noise_scheduler:
            noise_scheduler1.step()
        if has_grad_clip_scheduler:
            grad_clip_scheduler1.step()

        # load into set 2
        checkpoint_to_save = {"foo": "bar"}
        import io

        with io.BytesIO() as bytesio:
            pe1.save_checkpoint(
                path=bytesio,
                module=m1,
                optimizer=opt1,
                noise_scheduler=noise_scheduler1,
                grad_clip_scheduler=grad_clip_scheduler1,
                checkpoint_dict=checkpoint_to_save,
            )
            bytesio.seek(0)
            loaded_checkpoint = pe2.load_checkpoint(
                path=bytesio,
                module=m2,
                optimizer=opt2,
                noise_scheduler=noise_scheduler2,
                grad_clip_scheduler=grad_clip_scheduler2,
            )

        # check if loaded checkpoint has dummy dict
        self.assertTrue(
            "foo" in loaded_checkpoint and loaded_checkpoint["foo"] == "bar"
        )
        # check the two sets of components are now the same
        self.assertEqual(pe1.accountant.state_dict(), pe2.accountant.state_dict())
        self.assertTrue(are_state_dict_equal(m1.state_dict(), m2.state_dict()))
        if has_noise_scheduler:
            self.assertEqual(
                noise_scheduler1.state_dict(), noise_scheduler2.state_dict()
            )
        if has_grad_clip_scheduler:
            self.assertEqual(
                grad_clip_scheduler1.state_dict(), grad_clip_scheduler2.state_dict()
            )

        # check that non-state params are still different
        self.assertNotEqual(opt1.noise_multiplier, opt2.noise_multiplier)

    def test_validator_weight_update_check(self) -> None:
        """Test that privacy engine raises error if ModuleValidator.fix called after optimizer creation."""
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, 10), nn.Sigmoid())
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0, weight_decay=0
        )
        dl = self._init_data()
        model = ModuleValidator.fix(model)
        privacy_engine = PrivacyEngineGradSampleController()
        with self.assertRaisesRegex(
            ValueError, "Module parameters are different than optimizer Parameters"
        ):
            _, _, _ = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dl,
                noise_multiplier=1.1,
                max_grad_norm=1.0,
                grad_sample_mode=self.GRAD_SAMPLE_MODE,
            )

        # if optimizer is defined after ModuleValidator.fix() then raise no error
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0, weight_decay=0
        )
        controller, _, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            return_controller=True,
        )
        controller.cleanup()

    @given(
        noise_multiplier=st.floats(0.5, 5.0),
        max_steps=st.integers(3, 5),
        secure_mode=st.just(False),  # TODO: enable after fixing torchcsprng build
    )
    @settings(suppress_health_check=list(HealthCheck), deadline=None)
    def test_noise_level(
        self,
        noise_multiplier: float,
        max_steps: int,
        secure_mode: bool,
    ):
        """Test that the noise level is correctly set."""
        torch.manual_seed(100)
        # Initialize models with parameters to zero
        model, optimizer, dl, _ = self._init_private_training(
            noise_multiplier=noise_multiplier,
            secure_mode=secure_mode,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        for p in model.parameters():
            p.data.zero_()

        # Do max_steps steps of DP-SGD
        n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        steps = 0
        for x, _y in dl:
            optimizer.zero_grad()
            logits = model(x)
            loss = logits.view(logits.size(0), -1).sum(dim=1)
            # Gradient should be 0
            loss.backward(torch.zeros(logits.size(0)))

            optimizer.step()
            steps += 1

            if max_steps and steps >= max_steps:
                break

        # Noise should be equal to lr*sigma*sqrt(n_params * steps) / batch_size
        expected_norm = (
            steps
            * n_params
            * optimizer.noise_multiplier**2
            * self.LR**2
            / (optimizer.expected_batch_size**2)
        )
        real_norm = sum(
            [torch.sum(torch.pow(p.data, 2)) for p in model.parameters()]
        ).item()

        self.assertAlmostEqual(real_norm, expected_norm, delta=0.15 * expected_norm)

    @unittest.skip("requires torchcsprng compatible with new pytorch versions")
    @patch("torch.normal", MagicMock(return_value=torch.Tensor([0.6])))
    def test_generate_noise_in_secure_mode(self) -> None:
        """Test that noise is added correctly in secure_mode."""
        noise = _generate_noise(
            std=2.0,
            reference=torch.Tensor([1, 2, 3]),  # arbitrary size = 3
            secure_mode=True,
        )
        self.assertTrue(
            torch.allclose(noise, torch.Tensor([1.2, 1.2, 1.2])),
            "Model parameters after deterministic run must match",
        )


# ============================================================================
# Concrete Test Classes for Different Model Architectures
# ============================================================================


class SampleConvNet(nn.Module):
    """Simple ConvNet for testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.conv2(x)
        x = x.mean(dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PrivacyEngineGradSampleControllerConvNetGradSampleControllerTest(
    BasePrivacyEngineGradSampleControllerTest, unittest.TestCase
):
    """Test controller-based privacy engine with ConvNet."""

    def _init_data(self):
        ds = TensorDataset(
            torch.randn(self.DATA_SIZE, 1, 28, 28),
            torch.randint(low=0, high=10, size=(self.DATA_SIZE,)),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return SampleConvNet()


class SampleAttnNet(nn.Module):
    """LSTM + Attention model for testing."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 8)
        self.attn = DPMultiheadAttention(8, 1)
        self.fc = nn.Linear(8, 1)

        for param in self.parameters():
            nn.init.uniform_(param)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.attn(x, x, x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        return x


class MockTextDataset(Dataset):
    """Mock text dataset."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_first: bool = False):
        if batch_first:
            x_batch = x.shape[0]
        else:
            x_batch = x.shape[1]

        if x_batch != y.shape[0]:
            raise ValueError(
                f"Tensor shapes don't match. x:{x.shape}, y:{y.shape}, batch_first:{batch_first}"
            )

        self.x = x
        self.y = y
        self.batch_first = batch_first

    def __getitem__(self, index):
        if self.batch_first:
            return (self.x[index], self.y[index])
        else:
            return (self.x[:, index], self.y[index])

    def __len__(self):
        if self.batch_first:
            return self.x.shape[0]
        else:
            return self.x.shape[1]


def batch_second_collate(batch):
    """Collate function for batch-second data."""
    data = torch.stack([x[0] for x in batch]).permute(1, 0)
    labels = torch.stack([x[1] for x in batch])
    return data, labels


class PrivacyEngineGradSampleControllerTextGradSampleControllerTest(
    BasePrivacyEngineGradSampleControllerTest, unittest.TestCase
):
    """Test controller-based privacy engine with text/attention model."""

    def setUp(self):
        super().setUp()
        self.BATCH_FIRST = False

    def _init_data(self):
        x = torch.randint(0, 100, (12, self.DATA_SIZE))
        y = torch.randint(0, 12, (self.DATA_SIZE,))
        ds = MockTextDataset(x, y)
        return DataLoader(
            ds,
            batch_size=self.BATCH_SIZE,
            collate_fn=batch_second_collate,
            drop_last=False,
        )

    def _init_model(self):
        return SampleAttnNet()


class SampleTiedWeights(nn.Module):
    """Model with tied weights for testing."""

    def __init__(self, tie=True):
        super().__init__()
        self.emb = nn.Embedding(100, 8)
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 100)

        w = torch.empty(100, 8)
        nn.init.uniform_(w, -100, 100)

        if tie:
            p = nn.Parameter(w)
            self.emb.weight = p
            self.fc2.weight = p
        else:
            self.emb.weight = nn.Parameter(w.clone())
            self.fc2.weight = nn.Parameter(w.clone())

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.emb(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1)
        return x


class PrivacyEngineGradSampleControllerTiedWeightsGradSampleControllerTest(
    BasePrivacyEngineGradSampleControllerTest, unittest.TestCase
):
    """Test controller-based privacy engine with tied weights."""

    def _init_data(self):
        ds = TensorDataset(
            torch.randint(low=0, high=100, size=(self.DATA_SIZE,)),
            torch.randint(low=0, high=100, size=(self.DATA_SIZE,)),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return SampleTiedWeights(tie=True)


class SampleFrozenConvNet(nn.Module):
    """ConvNet with some frozen layers."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 10)

        # Freeze first conv layer
        for p in self.conv1.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.conv2(x)
        x = x.mean(dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class PrivacyEngineGradSampleControllerFrozenGradSampleControllerTest(
    BasePrivacyEngineGradSampleControllerTest, unittest.TestCase
):
    """Test controller-based privacy engine with frozen layers."""

    def _init_data(self):
        ds = TensorDataset(
            torch.randn(self.DATA_SIZE, 1, 28, 28),
            torch.randint(low=0, high=10, size=(self.DATA_SIZE,)),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return SampleFrozenConvNet()


# ============================================================================
# Functorch Variant Tests
# ============================================================================


class PrivacyEngineGradSampleControllerConvNetTestFunctorch(
    PrivacyEngineGradSampleControllerConvNetGradSampleControllerTest
):
    """Test controller-based privacy engine with ConvNet using functorch."""

    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class PrivacyEngineGradSampleControllerTextTestFunctorch(
    PrivacyEngineGradSampleControllerTextGradSampleControllerTest
):
    """Test controller-based privacy engine with text model using functorch."""

    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class PrivacyEngineGradSampleControllerTiedWeightsTestFunctorch(
    PrivacyEngineGradSampleControllerTiedWeightsGradSampleControllerTest
):
    """Test controller-based privacy engine with tied weights using functorch."""

    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class PrivacyEngineGradSampleControllerFrozenTestFunctorch(
    PrivacyEngineGradSampleControllerFrozenGradSampleControllerTest
):
    """Test controller-based privacy engine with frozen layers using functorch."""

    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


if __name__ == "__main__":
    unittest.main()
