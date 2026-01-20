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
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights
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
        # x of shape [B, 1, 28, 28]
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


class GradSampleModuleTest(unittest.TestCase):
    CLS = GradSampleModule

    def setUp(self):
        self.original_model = SampleConvNet()
        copy_of_original_model = SampleConvNet()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module = self.CLS(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
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

    def test_outputs_unaltered(self):
        """
        Test that boxing with GradSampleModule won't alter any outputs.
        Gradients are tested in the various `grad_samples` tests.
        """
        x, _ = next(iter(self.dl))
        self.original_model = self.original_model.eval()
        self.grad_sample_module = self.grad_sample_module.eval()
        with torch.no_grad():
            normal_out = self.original_model(x)
            gs_out = self.grad_sample_module(x)
        msg = (
            f"GradSample L2 norm = : {gs_out.norm(2)}, ",
            f"Original L2 norm = : {normal_out.norm(2)}, ",
            f"MSE = {F.mse_loss(gs_out, normal_out)}, ",
            f"L1 Loss = {F.l1_loss(gs_out, normal_out)}",
        )
        assert_close(gs_out, normal_out, atol=1e-7, rtol=1e-5, msg=msg)

    def test_zero_grad(self):
        x, _ = next(iter(self.dl))
        self.original_model = self.original_model.train()
        self.grad_sample_module = self.grad_sample_module.train()
        gs_out = self.grad_sample_module(x)
        loss = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss.backward()

        self.grad_sample_module.zero_grad()
        params_with_gs = [
            n
            for n, p in self.grad_sample_module.named_parameters()
            if p.grad_sample is not None
        ]
        msg = (
            "After calling .zero_grad() on the GradSampleModule, the following parameters still "
            f"have a grad_sample: {params_with_gs}"
        )
        assert len(params_with_gs) == 0, msg

    def test_to_standard_module(self):
        copy_of_original_model = SampleConvNet()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(),
            strict=True,
        )
        new_grad_sample_module = self.CLS(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )

        new_grad_sample_module = new_grad_sample_module.to_standard_module()

        assert isinstance(new_grad_sample_module, type(self.original_model))

        original_state_dict = self.original_model.state_dict()
        gs_state_dict = new_grad_sample_module.state_dict()

        missing_keys = gs_state_dict.keys() - original_state_dict.keys()
        assert not missing_keys, f"The following keys are missing: {missing_keys}"

        extra_keys = original_state_dict.keys() - gs_state_dict.keys()
        assert not extra_keys, f"The following keys are extra: {extra_keys}"

        for key in original_state_dict:
            original_tensor = original_state_dict[key].float()
            gs_tensor = gs_state_dict[key].float()
            msg = (
                f"Param {key}: GradSample L2 norm = : {gs_tensor.norm(2)}, ",
                f"Original L2 norm = : {original_tensor.norm(2)}, ",
                f"MSE = {F.mse_loss(gs_tensor, original_tensor)}, ",
                f"L1 Loss = {F.l1_loss(gs_tensor, original_tensor)}",
            )

            assert_close(gs_tensor, original_tensor, atol=1e-6, rtol=1e-4, msg=msg)

    def test_remove_hooks(self):
        """
        Test that after calling .remove_hooks() no hooks are left
        """
        copy_of_original_model = SampleConvNet()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(),
            strict=True,
        )
        new_grad_sample_module = GradSampleModule(
            copy_of_original_model, batch_first=True, loss_reduction="mean"
        )
        new_grad_sample_module.remove_hooks()

        remaining_forward_hooks = {
            module: module._forward_hooks
            for module in new_grad_sample_module.modules()
            if module._forward_hooks
        }
        assert (
            not remaining_forward_hooks
        ), f"Some forward hooks remain after .remove_hooks(): {remaining_forward_hooks}"

        remaining_backward_hooks = {
            module: module._backward_hooks
            for module in new_grad_sample_module.modules()
            if module._backward_hooks
        }
        assert (
            not remaining_backward_hooks
        ), f"Some backward hooks remain after .remove_hooks(): {remaining_backward_hooks}"

    def test_enable_hooks(self):
        self.grad_sample_module.enable_hooks()
        assert self.grad_sample_module.hooks_enabled

    def test_disable_hooks(self):
        self.grad_sample_module.disable_hooks()
        assert not self.grad_sample_module.hooks_enabled

    def test_standard_module_validation(self):
        class SimpleLinear(nn.Module):
            def __init__(self, in_f, out_f):
                super().__init__()
                self.p = nn.Parameter(torch.Tensor(in_f, out_f))

            def forward(self, x: torch.Tensor):
                return F.linear(x, self.p)

        # Should be handled by functorch
        try:
            gsm = GradSampleModule(SimpleLinear(4, 2))
            self.assertTrue(hasattr(gsm._module, "ft_compute_sample_grad"))
        except ImportError:
            print("Test could not be ran because functorch not available")

        # Should not raise exception if strict=False
        try:
            GradSampleModule(SimpleLinear(4, 2), strict=False)
        except ImportError:
            print("Test could not be ran because functorch not available")

        # Should not fail after relevant grad sampler has been registered
        register_grad_sampler(SimpleLinear)(compute_linear_grad_sample)
        GradSampleModule(SimpleLinear(4, 2))

    def test_custom_module_validation(self) -> None:
        with self.assertRaises(NotImplementedError):
            GradSampleModule(mobilenet_v3_small())

    def test_submodule_access(self) -> None:
        _ = self.grad_sample_module.fc1
        _ = self.grad_sample_module.fc2

        with self.assertRaises(AttributeError):
            _ = self.grad_sample_module.fc3

    def test_state_dict(self) -> None:
        gs_state_dict = self.grad_sample_module.state_dict()
        og_state_dict = self.original_model.state_dict()
        # check wrapped module state dict
        for key in og_state_dict.keys():
            self.assertTrue(f"_module.{key}" in gs_state_dict)
            assert_close(og_state_dict[key], gs_state_dict[f"_module.{key}"])

    def test_load_state_dict(self) -> None:
        gs_state_dict = self.grad_sample_module.state_dict()
        new_gs = GradSampleModule(
            SampleConvNet(), batch_first=False, loss_reduction="mean"
        )
        new_gs.load_state_dict(gs_state_dict)
        # wrapped module is the same
        for key in self.original_model.state_dict().keys():
            self.assertTrue(key in new_gs._module.state_dict())
            assert_close(
                self.original_model.state_dict()[key], new_gs._module.state_dict()[key]
            )


class EWGradSampleModuleTest(GradSampleModuleTest):
    CLS = GradSampleModuleExpandedWeights

    def test_remove_hooks(self) -> None:
        pass

    def test_enable_hooks(self) -> None:
        pass

    def test_disable_hooks(self) -> None:
        pass
