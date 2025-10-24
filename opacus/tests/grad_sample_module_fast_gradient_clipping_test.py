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
import logging
import unittest

import hypothesis.strategies as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from hypothesis import given, settings
from opacus.grad_sample import GradSampleModule, GradSampleModuleFastGradientClipping
from opacus.optimizers import DPOptimizer, DPOptimizerFastGradientClipping
from opacus.utils.fast_gradient_clipping_utils import DPLossFastGradientClipping
from opacus.utils.per_sample_gradients_utils import clone_module
from torch.utils.data import DataLoader, Dataset

from .grad_sample_module_test import GradSampleModuleTest, SampleConvNet


class SyntheticDataset(Dataset):
    def __init__(self, size, length, dim):
        self.size = size
        self.length = length
        self.dim = dim
        self.images = torch.randn(self.size, self.length, self.dim)
        self.labels = torch.randint(
            0, 2, size=(self.size, self.length), dtype=torch.float32
        )

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


class SampleModule(nn.Module):
    def __init__(self):
        super(SampleModule, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc3 = nn.Linear(2, 1024)
        self.fc4 = nn.Linear(1024, 10)
        self.fc5 = nn.Linear(10, 1)
        self.layer_norm = nn.LayerNorm(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x).flatten(start_dim=1)
        x = F.softmax(x)
        return x


class SampleEmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SampleEmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Manually set weights for the embedding layer for testing
        self.embedding.weight = nn.Parameter(torch.tensor([[0.1], [0.2], [0.3]]))

    def forward(self, x):
        x = self.embedding(x)
        return x


class GradSampleModuleFastGradientClippingTest(GradSampleModuleTest):
    CLS = GradSampleModuleFastGradientClipping

    def setUp(self):
        self.dim = 2
        self.size = 10
        self.length = 5
        # self.original_model = SampleModule()
        # copy_of_original_model = SampleModule()

        self.original_model = SampleConvNet()
        copy_of_original_model = SampleConvNet()

        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module = self.CLS(
            copy_of_original_model,
            batch_first=True,
            max_grad_norm=1,
            use_ghost_clipping=True,
        )
        self.DATA_SIZE = self.size
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data_sequantial(self, size, length, dim):
        self.size = size
        self.length = length
        self.dim = dim
        dataset = SyntheticDataset(size=size, length=length, dim=dim)
        self.dl = DataLoader(dataset, batch_size=size, shuffle=True)

    @given(
        size=st.sampled_from([10]),
        length=st.sampled_from([5]),
        dim=st.sampled_from([2]),
    )
    @settings(deadline=1000000)
    def test_norm_calculation_fast_gradient_clipping(self, size, length, dim):
        """
        Tests if norm calculation is the same between standard (opacus) and fast gradient clipping"
        """
        torch.set_default_dtype(torch.float64)

        self.length = length
        self.size = size
        self.dim = dim

        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        self.setUp_data_sequantial(self.size, self.length, self.dim)
        noise_multiplier = 0.0
        batch_size = self.size
        max_grad_norm = 1.0
        sample_module = SampleModule()
        self.model_normal = GradSampleModule(clone_module(sample_module))
        optimizer_normal = torch.optim.SGD(self.model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        self.grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
            loss_reduction="mean",
        )
        optimizer_gc = torch.optim.SGD(self.grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
            loss_reduction="mean",
        )

        (input_data, target_data) = list(self.dl)[0]
        optimizer_normal.zero_grad()
        output_normal = self.model_normal(input_data)
        loss_normal = self.criterion(output_normal, target_data)
        loss_normal.backward()
        all_norms_normal = torch.stack(
            [
                torch.stack([g.norm() for g in param.grad_sample], dim=0)
                for param in self.model_normal.parameters()
            ],
            dim=0,
        )
        flat_norms_normal = torch.cat([p.flatten() for p in all_norms_normal])

        optimizer_gc.zero_grad()
        criterion_gc = DPLossFastGradientClipping(
            self.grad_sample_module, optimizer_gc, copy.deepcopy(self.criterion)
        )
        output_gc = self.grad_sample_module(input_data)
        loss_gc = criterion_gc(output_gc, target_data)
        loss_gc.backward()

        all_norms_gc = [
            param._norm_sample for param in self.grad_sample_module.parameters()
        ]
        flat_norms_gc = torch.cat([p.flatten() for p in all_norms_gc])

        diff = flat_norms_normal - flat_norms_gc

        logging.info(f"Max difference between (vanilla) Opacus and FGC = {max(diff)}")
        msg = "Fail: Per-sample gradient norms from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_norms_normal, flat_norms_gc, atol=1e-3), msg

    @given(
        size=st.sampled_from([10]),
        length=st.sampled_from([5]),
        dim=st.sampled_from([2]),
    )
    @settings(deadline=1000000)
    def test_weight_update_fast_gradient_clipping(self, size, length, dim):
        """
        Tests if weight updates are the same between standard (opacus) and fast gradient clipping"
        """
        torch.set_default_dtype(torch.float32)

        noise_multiplier = 0.0
        batch_size = size
        self.length = length
        self.size = size
        self.dim = dim
        self.setUp_data_sequantial(self.size, self.length, self.dim)
        max_grad_norm = 1.0
        self.criterion = torch.nn.CrossEntropyLoss()

        sample_module = SampleModule()
        self.model_normal = GradSampleModule(clone_module(sample_module))
        self.grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )

        optimizer_normal = torch.optim.SGD(self.model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        optimizer_gc = torch.optim.SGD(self.grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        criterion_gc = DPLossFastGradientClipping(
            self.grad_sample_module, optimizer_gc, copy.deepcopy(self.criterion)
        )

        (input_data, target_data) = list(self.dl)[0]
        optimizer_normal.zero_grad()
        output_normal = self.model_normal(input_data)
        loss_normal = torch.mean(self.criterion(output_normal, target_data), dim=0)
        loss_normal.backward()
        optimizer_normal.step()

        all_params_normal = [param for param in self.model_normal.parameters()]
        flat_params_normal = torch.cat([p.flatten() for p in all_params_normal])

        output_gc = self.grad_sample_module(input_data)

        loss_gc = criterion_gc(output_gc, target_data)
        loss_gc.backward()
        optimizer_gc.step()

        all_params_gc = [param for param in self.grad_sample_module.parameters()]
        flat_params_gc = torch.cat([p.flatten() for p in all_params_gc])

        diff = torch.tensor(
            [
                (p_gc - p_normal).norm()
                for (p_gc, p_normal) in zip(flat_params_gc, flat_params_normal)
            ]
        )
        logging.info(
            f"Max difference between the model parameters of (vanilla) Opacus and FGC = {max(diff)}"
        )
        msg = "Fail: Parameters from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_params_normal, flat_params_gc, atol=1e-3), msg


class GradSampleModuleFastGradientClippingEmbeddingLayerTest(unittest.TestCase):

    def test_norm_calculation(self):
        """
        Tests if norm calculation for embedding layer is the same between
        standard (Opacus) and fast gradient clipping"
        """
        vocab_size = 3
        embedding_dim = 1

        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        noise_multiplier = 0.0
        input_data = torch.tensor([[1, 1], [2, 0], [2, 0]], dtype=torch.long)
        batch_size = 3
        max_grad_norm = 1.0
        sample_module = SampleEmbeddingModule(vocab_size, embedding_dim)
        model_normal = GradSampleModule(clone_module(sample_module))
        optimizer_normal = torch.optim.SGD(model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )
        optimizer_gc = torch.optim.SGD(grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        optimizer_normal.zero_grad()
        output_normal = model_normal(input_data)
        target_data = torch.rand_like(output_normal)

        loss_normal = torch.mean(criterion(output_normal, target_data), dim=0)
        loss_normal.backward()
        all_norms_normal = torch.stack(
            [
                torch.stack([g.norm() for g in param.grad_sample], dim=0)
                for param in model_normal.parameters()
            ],
            dim=0,
        )
        flat_norms_normal = torch.cat([p.flatten() for p in all_norms_normal])

        grad_sample_module.enable_hooks()
        output_gc = grad_sample_module(input_data)

        first_loss_per_sample = criterion(output_gc, target_data)
        first_loss = torch.mean(first_loss_per_sample)
        first_loss.backward(retain_graph=True)

        optimizer_gc.zero_grad()
        coeff = grad_sample_module.get_clipping_coef()
        second_loss_per_sample = coeff * first_loss_per_sample
        second_loss = torch.sum(second_loss_per_sample)
        grad_sample_module.disable_hooks()
        second_loss.backward()

        all_norms_gc = [param._norm_sample for param in grad_sample_module.parameters()]
        flat_norms_gc = torch.cat([p.flatten() for p in all_norms_gc])

        diff = flat_norms_normal - flat_norms_gc

        logging.info(f"Max difference between (vanilla) Opacus and FGC = {max(diff)}")
        msg = "Fail: Gradient norms from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_norms_normal, flat_norms_gc, atol=1e-3), msg

    def test_gradient_calculation(self):
        """Tests if gradients for embedding layer are the same between standard
        (Opacus) and fast gradient clipping."""

        noise_multiplier = 0.0
        vocab_size = 3
        embedding_dim = 1
        batch_size = 3
        input_data = torch.tensor([[1, 1], [2, 0], [2, 0]], dtype=torch.long)
        max_grad_norm = 1.0
        criterion = torch.nn.CrossEntropyLoss()

        sample_module = SampleEmbeddingModule(vocab_size, embedding_dim)
        model_normal = GradSampleModule(clone_module(sample_module))
        grad_sample_module = GradSampleModuleFastGradientClipping(
            clone_module(sample_module),
            max_grad_norm=max_grad_norm,
            use_ghost_clipping=True,
        )

        optimizer_normal = torch.optim.SGD(model_normal.parameters(), lr=1)
        optimizer_normal = DPOptimizer(
            optimizer_normal,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        optimizer_gc = torch.optim.SGD(grad_sample_module.parameters(), lr=1)
        optimizer_gc = DPOptimizerFastGradientClipping(
            optimizer_gc,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=batch_size,
        )

        criterion_gc = DPLossFastGradientClipping(
            grad_sample_module, optimizer_gc, criterion
        )

        optimizer_normal.zero_grad()
        output_normal = model_normal(input_data)
        target_data = torch.tensor([[[0.1], [0.1]], [[0.2], [0.3]], [[0.2], [0.3]]])
        loss_normal = torch.mean(criterion(output_normal, target_data), dim=0)
        loss_normal.backward()
        optimizer_normal.step()

        all_grads_normal = [param.summed_grad for param in model_normal.parameters()]
        flat_grads_normal = torch.cat([p.flatten() for p in all_grads_normal])

        optimizer_gc.zero_grad()
        grad_sample_module.enable_hooks()
        output_gc = grad_sample_module(input_data)

        loss_gc = criterion_gc(output_gc, target_data)
        loss_gc.backward()
        optimizer_gc.step()

        all_grads_gc = [param.grad for param in grad_sample_module.parameters()]
        flat_grads_gc = torch.cat([p.flatten() for p in all_grads_gc])
        diff = torch.tensor(
            [
                (g_gc - g_normal).norm()
                for (g_gc, g_normal) in zip(flat_grads_gc, flat_grads_normal)
            ]
        )

        logging.info(f"Max difference between (vanilla) Opacus and FGC = {max(diff)}")
        msg = "Fail: Gradients from vanilla DP-SGD and from fast gradient clipping are different"
        assert torch.allclose(flat_grads_normal, flat_grads_gc, atol=1e-3), msg

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_multidevice_get_norm_sample(self):
        """Test that get_norm_sample handles parameters on different devices."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        # Create a simple model with parameters on different devices
        class MultiDeviceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20).to(device1)
                self.fc2 = nn.Linear(20, 5).to(device2)

            def forward(self, x):
                x = x.to(device1)
                x = torch.relu(self.fc1(x))
                x = x.to(device2)
                return self.fc2(x)

        model = MultiDeviceModel()
        grad_sample_module = GradSampleModuleFastGradientClipping(
            model, max_grad_norm=1.0, use_ghost_clipping=False
        )

        # Simulate _norm_sample on different devices
        batch_size = 4
        for param in grad_sample_module.trainable_parameters:
            param._norm_sample = torch.randn(batch_size, device=param.device)

        # This should not raise any device mismatch errors
        try:
            norm_sample = grad_sample_module.get_norm_sample()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in get_norm_sample: {e}")
            else:
                raise

        self.assertTrue(
            success, "get_norm_sample should handle multi-device parameters"
        )
        self.assertEqual(norm_sample.shape[0], batch_size)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_multidevice_get_clipping_coef(self):
        """Test that get_clipping_coef handles parameters on different devices."""
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")

        # Create a simple model with parameters on different devices
        class MultiDeviceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 20).to(device1)
                self.fc2 = nn.Linear(20, 5).to(device2)

            def forward(self, x):
                x = x.to(device1)
                x = torch.relu(self.fc1(x))
                x = x.to(device2)
                return self.fc2(x)

        model = MultiDeviceModel()
        max_grad_norm = 1.0
        grad_sample_module = GradSampleModuleFastGradientClipping(
            model, max_grad_norm=max_grad_norm, use_ghost_clipping=False
        )

        # Simulate _norm_sample on different devices
        batch_size = 4
        for param in grad_sample_module.trainable_parameters:
            # Create norms with values that will require clipping
            param._norm_sample = torch.ones(batch_size, device=param.device) * 2.0

        # This should not raise any device mismatch errors
        try:
            clipping_coef = grad_sample_module.get_clipping_coef()
            success = True
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                success = False
                self.fail(f"Device mismatch error in get_clipping_coef: {e}")
            else:
                raise

        self.assertTrue(
            success, "get_clipping_coef should handle multi-device parameters"
        )
        self.assertEqual(clipping_coef.shape[0], batch_size)
        # Verify clipping coefficients are correct
        self.assertTrue(torch.all(clipping_coef <= 1.0))
