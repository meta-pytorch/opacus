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

import abc
import io
import itertools
import math
import unittest
from abc import ABC
from typing import Optional, OrderedDict
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings
from opacus import PrivacyEngine
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from opacus.optimizers.optimizer import _generate_noise
from opacus.schedulers import StepGradClip, StepNoise
from opacus.utils.module_utils import are_state_dict_equal
from opacus.validators.errors import UnsupportedModuleError
from opacus.validators.module_validator import ModuleValidator
from opt_einsum import contract
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import models, transforms
from torchvision.datasets import FakeData

from .utils import CustomLinearModule, LinearWithExtraParam


def get_grad_sample_aggregated(tensor: torch.Tensor, loss_type: str = "mean"):
    if tensor.grad_sample is None:
        raise ValueError(
            f"The input tensor {tensor} has grad computed, but missing grad_sample."
            f"Please attach PrivacyEngine"
        )

    if loss_type not in ("sum", "mean"):
        raise ValueError(f"loss_type = {loss_type}. Only 'sum' and 'mean' supported")

    grad_sample_aggregated = contract("i...->...", tensor.grad_sample)
    if loss_type == "mean":
        b_sz = tensor.grad_sample.shape[0]
        grad_sample_aggregated /= b_sz

    return grad_sample_aggregated


class BasePrivacyEngineTest(ABC):
    def setUp(self):
        self.DATA_SIZE = 512
        self.BATCH_SIZE = 64
        self.LR = 0.5
        self.ALPHAS = [1 + x / 10.0 for x in range(1, 100, 10)]
        self.criterion = nn.CrossEntropyLoss()
        self.BATCH_FIRST = True
        self.GRAD_SAMPLE_MODE = "hooks"

        torch.manual_seed(42)

    @abc.abstractmethod
    def _init_data(self) -> None:
        pass

    @abc.abstractmethod
    def _init_model(self) -> None:
        pass

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
        model = PrivacyEngine.get_compatible_module(model)
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

        privacy_engine = PrivacyEngine(secure_mode=secure_mode)
        model, optimizer, poisson_dl = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_first=self.BATCH_FIRST,
            poisson_sampling=poisson_sampling,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
        )

        return model, optimizer, poisson_dl, privacy_engine

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

    def _train_steps_with_closure(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        dl: DataLoader,
        max_steps: Optional[int] = None,
    ):
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

    def test_basic(self) -> None:
        for opt_exclude_frozen in [True, False]:
            with self.subTest(opt_exclude_frozen=opt_exclude_frozen):
                model, optimizer, dl, _ = self._init_private_training(
                    noise_multiplier=1.0,
                    max_grad_norm=1.0,
                    poisson_sampling=True,
                    grad_sample_mode=self.GRAD_SAMPLE_MODE,
                    opt_exclude_frozen=opt_exclude_frozen,
                )
                self._train_steps(model, optimizer, dl)

    def _compare_to_vanilla(
        self,
        do_noise,
        do_clip,
        expected_match,
        grad_sample_mode,
        use_closure=False,
        max_steps=1,
    ):
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
        """
        Compare gradients and updated weights with vanilla model initialized
        with the same seed
        """
        self._compare_to_vanilla(
            do_noise=do_noise,
            do_clip=do_clip,
            expected_match=not (do_noise or do_clip),
            use_closure=use_closure,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
            max_steps=max_steps,
        )

    def test_flat_clipping(self) -> None:
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

    def test_per_layer_clipping(self) -> None:
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

    def test_sample_grad_aggregation(self) -> None:
        """
        Check if final gradient is indeed an aggregation over per-sample gradients
        """
        model, optimizer, dl, _ = self._init_private_training(
            poisson_sampling=False,
            noise_multiplier=0.0,
            max_grad_norm=99999.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        self._train_steps(model, optimizer, dl, max_steps=1)

        for p_name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            summed_grad = p.grad_sample.sum(dim=0) / self.BATCH_SIZE
            self.assertTrue(
                torch.allclose(p.grad, summed_grad, atol=1e-8, rtol=1e-4),
                f"Per sample gradients don't sum up to the final grad value."
                f"Param: {p_name}",
            )

    def test_noise_changes_every_time(self) -> None:
        """
        Test that adding noise results in ever different model params.
        We disable clipping in this test by setting it to a very high threshold.
        """
        model, optimizer, dl, _ = self._init_private_training(
            poisson_sampling=False, grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(model, optimizer, dl, max_steps=1)
        first_run_params = (p for p in model.parameters() if p.requires_grad)

        model, optimizer, dl, _ = self._init_private_training(
            poisson_sampling=False, grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(model, optimizer, dl, max_steps=1)
        second_run_params = (p for p in model.parameters() if p.requires_grad)

        for p0, p1 in zip(first_run_params, second_run_params):
            self.assertFalse(torch.allclose(p0, p1))

    def test_get_compatible_module_inaction(self) -> None:
        needs_no_replacement_module = nn.Linear(1, 2)
        fixed_module = PrivacyEngine.get_compatible_module(needs_no_replacement_module)
        self.assertFalse(fixed_module is needs_no_replacement_module)
        self.assertTrue(
            are_state_dict_equal(
                needs_no_replacement_module.state_dict(), fixed_module.state_dict()
            )
        )

    def test_model_validator(self) -> None:
        """
        Test that the privacy engine raises errors
        if there are unsupported modules
        """
        resnet = models.resnet18()
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        dl = self._init_data()
        privacy_engine = PrivacyEngine()
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
        """
        Test that the privacy engine fixes unsupported modules
        and succeeds.
        """
        resnet = PrivacyEngine.get_compatible_module(models.resnet18())
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1.0)
        dl = self._init_data()
        privacy_engine = PrivacyEngine()
        _, _, _ = privacy_engine.make_private(
            module=resnet,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.3,
            max_grad_norm=1,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        self.assertTrue(1, 1)

    def test_make_private_with_epsilon(self) -> None:
        model, optimizer, dl = self._init_vanilla_training()
        target_eps = 2.0
        target_delta = 1e-5
        epochs = 2
        total_steps = epochs * len(dl)

        privacy_engine = PrivacyEngine()
        model, optimizer, poisson_dl = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            target_epsilon=target_eps,
            target_delta=1e-5,
            epochs=epochs,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )
        self._train_steps(model, optimizer, poisson_dl, max_steps=total_steps)
        self.assertAlmostEqual(
            target_eps, privacy_engine.get_epsilon(target_delta), places=2
        )

    def test_deterministic_run(self) -> None:
        """
        Tests that for 2 different models, secure seed can be fixed
        to produce same (deterministic) runs.
        """
        torch.manual_seed(0)
        m1, opt1, dl1, _ = self._init_private_training(
            grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(m1, opt1, dl1)
        params1 = [p for p in m1.parameters() if p.requires_grad]

        torch.manual_seed(0)
        m2, opt2, dl2, _ = self._init_private_training(
            grad_sample_mode=self.GRAD_SAMPLE_MODE
        )
        self._train_steps(m2, opt2, dl2)
        params2 = [p for p in m2.parameters() if p.requires_grad]

        for p1, p2 in zip(params1, params2):
            self.assertTrue(
                torch.allclose(p1, p2),
                "Model parameters after deterministic run must match",
            )

    def test_validator_weight_update_check(self) -> None:
        """
        Test that the privacy engine raises error if ModuleValidator.fix(model) is
        called after the optimizer is created
        """
        model = models.densenet121(weights=None)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Linear(num_ftrs, 10), nn.Sigmoid())
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0, weight_decay=0
        )
        dl = self._init_data()
        model = ModuleValidator.fix(model)
        privacy_engine = PrivacyEngine()
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
        _, _, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dl,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            grad_sample_mode=self.GRAD_SAMPLE_MODE,
        )

    def test_parameters_match(self) -> None:
        dl = self._init_data()

        m1 = self._init_model()
        m2 = self._init_model()
        m2.load_state_dict(m1.state_dict())
        # optimizer is initialized with m2 parameters
        opt = torch.optim.SGD(m2.parameters(), lr=0.1)

        # the values are the identical
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

        privacy_engine = PrivacyEngine()
        # but model parameters and optimzier parameters must be the same object,
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
    def test_checkpoints(
        self, has_noise_scheduler: bool, has_grad_clip_scheduler: bool
    ):
        # 1. Disable poisson sampling to avoid randomness in data loading caused by changing seeds.
        # 2. Use noise_multiplier=0.0 to avoid randomness in torch.normal()
        # create a set of components: set 1
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
        self._train_steps(m1, opt1, dl1)
        if has_noise_scheduler:
            noise_scheduler1.step()
        if has_grad_clip_scheduler:
            grad_clip_scheduler1.step()

        # load into set 2
        checkpoint_to_save = {"foo": "bar"}
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
        """
        Tests that the noise level is correctly set
        """

        def helper_test_noise_level(
            noise_multiplier: float, max_steps: int, secure_mode: bool
        ):
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

        helper_test_noise_level(
            noise_multiplier=noise_multiplier,
            max_steps=max_steps,
            secure_mode=secure_mode,
        )

    @unittest.skip("requires torchcsprng compatible with new pytorch versions")
    @patch("torch.normal", MagicMock(return_value=torch.Tensor([0.6])))
    def test_generate_noise_in_secure_mode(self) -> None:
        """
        Tests that the noise is added correctly in secure_mode,
        according to section 5.1 in https://arxiv.org/abs/2107.10138.
        Since n=2, noise should be summed 4 times and divided by 2.

        In this example, torch.normal returns a constant value of 0.6.
        So, the overal noise would be (0.6 + 0.6 + 0.6 + 0.6)/2 = 1.2
        """
        noise = _generate_noise(
            std=2.0,
            reference=torch.Tensor([1, 2, 3]),  # arbitrary size = 3
            secure_mode=True,
        )
        self.assertTrue(
            torch.allclose(noise, torch.Tensor([1.2, 1.2, 1.2])),
            "Model parameters after deterministic run must match",
        )


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 3)
        self.gnorm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv1d(16, 32, 3, 1)
        self.lnorm1 = nn.LayerNorm((32, 23))
        self.conv3 = nn.Conv1d(32, 32, 3, 1, bias=False)
        # self.instnorm1 = nn.InstanceNorm1d(32, affine=True)
        self.convf = nn.Conv1d(32, 32, 1, 1)
        for p in self.convf.parameters():
            p.requires_grad = False
        self.fc1 = nn.Linear(21, 17)
        self.lnorm2 = nn.LayerNorm(17)
        self.fc2 = nn.Linear(32 * 17, 10)

        # for layer in (self.gnorm1, self.lnorm1, self.lnorm2, self.instnorm1):
        #     nn.init.uniform_(layer.weight)
        #     nn.init.uniform_(layer.bias)
        for param in self.parameters():
            nn.init.uniform_(param)

    def forward(self, x):
        # x of shape [B, 1, 28, 28]
        x = self.conv1(x)  # -> [B, 16, 10, 10]
        x = self.gnorm1(x)  # -> [B, 16, 10, 10]
        x = F.relu(x)  # -> [B, 16, 10, 10]
        x = F.max_pool2d(x, 2, 2)  # -> [B, 16, 5, 5]
        x = x.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])  # -> [B, 16, 25]
        x = self.conv2(x)  # -> [B, 32, 23]
        x = self.lnorm1(x)  # -> [B, 32, 23]
        x = F.relu(x)  # -> [B, 32, 23]
        x = self.conv3(x)  # -> [B, 32, 21]
        # x = self.instnorm1(x)  # -> [B, 32, 21]
        x = self.convf(x)  # -> [B, 32, 21]
        x = self.fc1(x)  # -> [B, 32, 17]
        x = self.lnorm2(x)  # -> [B, 32, 17]
        x = x.view(-1, x.shape[-2] * x.shape[-1])  # -> [B, 32 * 17]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"


class PrivacyEngineConvNetTest(BasePrivacyEngineTest, unittest.TestCase):
    def _init_data(self):
        ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return SampleConvNet()


@pytest.mark.skip(("Incompatible with the new empty batch handling"))
class PrivacyEngineConvNetEmptyBatchTest(PrivacyEngineConvNetTest):
    def setUp(self) -> None:
        super().setUp()

        # This will trigger multiple empty batches with poisson sampling enabled
        self.BATCH_SIZE = 1

    def test_checkpoints(self) -> None:
        pass

    def test_noise_level(self) -> None:
        pass


class PrivacyEngineConvNetFrozenTest(BasePrivacyEngineTest, unittest.TestCase):
    def _init_data(self):
        ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(1, 35, 35),
            num_classes=10,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        m = SampleConvNet()
        for p in itertools.chain(m.conv1.parameters(), m.gnorm1.parameters()):
            p.requires_grad = False

        return m


class PrivacyEngineConvNetFrozenTestFunctorch(PrivacyEngineConvNetFrozenTest):
    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class PrivacyEngineConvNetTestExpandedWeights(PrivacyEngineConvNetTest):
    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "ew"

    @unittest.skip("Original p.grad is not available in ExpandedWeights")
    def test_sample_grad_aggregation(self) -> None:
        pass


class PrivacyEngineConvNetTestFunctorch(PrivacyEngineConvNetTest):
    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class SampleAttnNet(nn.Module):
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
            return (
                self.x[
                    :,
                    index,
                ],
                self.y[index],
            )

    def __len__(self):
        return self.y.shape[0]


def batch_second_collate(batch):
    data = torch.stack([x[0] for x in batch]).permute(1, 0)
    labels = torch.stack([x[1] for x in batch])
    return data, labels


class PrivacyEngineTextTest(BasePrivacyEngineTest, unittest.TestCase):
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

    def _init_model(
        self, private=False, state_dict=None, model=None, **privacy_engine_kwargs
    ):
        return SampleAttnNet()


class PrivacyEngineTextTestFunctorch(PrivacyEngineTextTest):
    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class SampleTiedWeights(nn.Module):
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


class PrivacyEngineTiedWeightsTest(BasePrivacyEngineTest, unittest.TestCase):
    def setUp(self):
        super().setUp()

    def _init_data(self):
        ds = TensorDataset(
            torch.randint(low=0, high=100, size=(self.DATA_SIZE,)),
            torch.randint(low=0, high=100, size=(self.DATA_SIZE,)),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return SampleTiedWeights(tie=True)


class PrivacyEngineTiedWeightsTestFunctorch(PrivacyEngineTiedWeightsTest):
    def setUp(self) -> None:
        super().setUp()
        self.GRAD_SAMPLE_MODE = "functorch"


class ModelWithCustomLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = CustomLinearModule(4, 8)
        self.fc2 = LinearWithExtraParam(8, 4)
        self.extra_param = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.matmul(self.extra_param)
        return x


class PrivacyEngineCustomLayerTest(BasePrivacyEngineTest, unittest.TestCase):
    def _init_data(self):
        ds = TensorDataset(
            torch.randn(self.DATA_SIZE, 4),
            torch.randint(low=0, high=3, size=(self.DATA_SIZE,)),
        )
        return DataLoader(ds, batch_size=self.BATCH_SIZE, drop_last=False)

    def _init_model(self):
        return ModelWithCustomLinear()
