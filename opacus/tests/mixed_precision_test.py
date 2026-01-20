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
from typing import Dict, Optional, Type

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.utils.data import DataLoader

from .mixed_precision_utils import (
    AttentionModel,
    ComplexModel,
    Conv1DModel,
    Conv2DModel,
    Conv3DModel,
    EmbeddingBagModel,
    EmbeddingModel,
    RNNModel,
    SimpleLinearModel,
    SimpleLinearReluModel,
    create_random_data,
)


class MixedPrecisionTest(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def setUp(self):

        # Since this test only runs when CUDA is available, always use CUDA
        self.device = torch.device("cuda")
        self.input_dim = 4
        self.hidden_dim = 16
        self.output_dim = 4
        self.seq_len = 4
        self.batch_size = 2
        self.num_batches = 2

        # Check if bfloat16 is supported
        self.bf16_supported = hasattr(torch, "bfloat16")

    def _get_training_components(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        grad_sample_mode: str,
        dtype: torch.dtype,
        wrap_model: bool = True,
    ):
        """
        Return training components (model, optimizer, criterion, dataloader) wrapped by PrivacyEngine.
        """
        model = model.to(self.device)
        model = model.to(dtype)
        model = model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        privacy_engine = PrivacyEngine()

        # Make the model private with the specified precision
        if grad_sample_mode in ["hooks", "functorch", "ew"]:
            hooks_or_wrapper, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                noise_multiplier=1,
                max_grad_norm=1,
                grad_sample_mode=grad_sample_mode,
                poisson_sampling=False,
                wrap_model=wrap_model,
            )
        elif grad_sample_mode == "ghost":
            hooks_or_wrapper, optimizer, criterion, dataloader = (
                privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=dataloader,
                    criterion=criterion,
                    max_grad_norm=1,
                    noise_multiplier=1,
                    grad_sample_mode="ghost",
                    poisson_sampling=False,
                    wrap_model=wrap_model,
                )
            )

        if wrap_model:
            model = hooks_or_wrapper

        return model, optimizer, criterion, dataloader

    def _train_mixed_precision(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dtype: torch.dtype,
        grad_sample_mode: str = "hooks",
        wrap_model: bool = True,
    ):
        """
        Integration test for training a model with mixed precison (FP32+FP16 or FP32+BF16).
        It checks dtypes of various training artifacts.
        The expected behavior is that:
            - model parameters are in full precision FP32
            - model outputs are in low precision (BF16 or FP16)
            - gradients are in high precision (FP32)

        Args:
            model (nn.Module): The neural network model to be trained.
            dataloader (DataLoader): DataLoader providing the training data.
            dtype (torch.dtype): The lower data type for mixed precision training (torch.float16 or torch.bfloat16).
            grad_sample_mode (str): The mode for per-sample gradient computation, options include "hooks", "functorch", "ew", and "ghost".
            wrap_model (bool): Whether to wrap the model or use hooks.
        """

        model, optimizer, criterion, dataloader = self._get_training_components(
            model,
            dataloader,
            grad_sample_mode,
            dtype=torch.float32,
            wrap_model=wrap_model,
        )
        # model weights should be in high precision (fp32)
        for p in model.parameters():
            self.assertTrue(p.dtype == torch.float32)

        for batch in dataloader:
            x, y = batch
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=dtype):
                outputs = model(x)
                self.assertTrue(outputs.dtype == dtype)

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                # the gradients should have been cast up to high precision (fp32)
                if p.grad is not None:
                    self.assertTrue(p.grad.dtype == torch.float32)
                # grad_sample and norm_sample could be either in FP32 or low precision depending on the parameter
                # we do not explicitly cast them up to FP32, we only ensure that final gradients are cast up
                if p.grad_sample is not None:
                    self.assertTrue(p.grad_sample.dtype in [torch.float32, dtype])
                if grad_sample_mode == "ghost" and p._norm_sample is not None:
                    self.assertTrue(p._norm_sample.dtype in [torch.float32, dtype])

            optimizer.step()

    def _train_low_precision(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        dtype: torch.dtype,
        grad_sample_mode: str = "hooks",
        wrap_model: bool = True,
    ):
        """
        Runs an integration test for low precision training (BF16 or FP16).
        Tests that model weights, outputs, and gradients are in the low precision dtype.

        Args:
            model (nn.Module): The neural network model to be trained.
            dataloader (DataLoader): DataLoader providing the training data.
            dtype (torch.dtype): The data type for low precision training (torch.float16 or torch.bfloat16).
            grad_sample_mode (str): The mode for per-sample gradient computation, options include "hooks", "functorch", "ew", and "ghost".
            wrap_model (bool): Whether to wrap the model or use hooks.
        """

        model, optimizer, criterion, dataloader = self._get_training_components(
            model, dataloader, grad_sample_mode, dtype=dtype, wrap_model=wrap_model
        )

        for p in model.parameters():
            self.assertTrue(p.dtype == dtype)

        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            if x.is_floating_point():  # for embedding layers, keep input as int
                x = x.to(dtype)
            outputs = model(x)
            self.assertTrue(outputs.dtype == dtype)

            loss = criterion(outputs, y)
            loss.backward()

            # all gradients and gradient-related attributes should be in low precision
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(p.grad.dtype == dtype)
                if p.grad_sample is not None:
                    self.assertTrue(p.grad_sample.dtype == dtype)
                if grad_sample_mode == "ghost" and p._norm_sample is not None:
                    self.assertTrue(p._norm_sample.dtype == dtype)

            optimizer.step()

    def _test_precision_training(
        self,
        model_class: Type[nn.Module],
        model_kwargs: Optional[Dict] = None,
    ):
        """
        Integration tests for training a model with different precision settings: mixed and low precision.
        It tests several layer types and architectures with all grad sample modes.
        In particular, all layers implemented in Opacus are tested.
        The test checks that model weights, outputs, and gradients are in the expected dtypes.
        """
        if model_kwargs is None:
            model_kwargs = {}

        # Create random data
        dataloader, _ = create_random_data(
            model_class,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_batches=self.num_batches,
            seq_len=self.seq_len,
            device=self.device,
        )

        low_precision_type = [torch.float16]
        if self.bf16_supported:
            low_precision_type.append(torch.bfloat16)

        self._test_low_precision_all_modes(
            model_class, model_kwargs, dataloader, low_precision_type
        )
        self._test_mixed_precision_all_modes(
            model_class, model_kwargs, dataloader, low_precision_type
        )

    def _test_low_precision_all_modes(
        self, model_class, model_kwargs, dataloader, low_precision_type
    ):
        # Test with low precision
        for grad_sample_mode in ["hooks", "ghost", "functorch", "ew"]:
            for wrap_model in [True, False]:
                if not wrap_model and grad_sample_mode == "ew":
                    continue

                for dtype in low_precision_type:
                    with self.subTest(
                        grad_sample_mode=grad_sample_mode,
                        wrap_model=wrap_model,
                        dtype=dtype,
                    ):
                        # skip test for models with layers not supported by ew
                        if grad_sample_mode == "ew" and model_class in [
                            SimpleLinearModel,
                            EmbeddingModel,
                            EmbeddingBagModel,
                            ComplexModel,
                        ]:
                            continue
                        # functorch does not support EmbeddingBagModel
                        if (
                            grad_sample_mode == "functorch"
                            and model_class == EmbeddingBagModel
                        ):
                            continue
                        self._train_low_precision(
                            model=model_class(**model_kwargs),  # Create a fresh model
                            dataloader=dataloader,
                            dtype=dtype,
                            grad_sample_mode=grad_sample_mode,
                            wrap_model=wrap_model,
                        )

    def _test_mixed_precision_all_modes(
        self, model_class, model_kwargs, dataloader, low_precision_type
    ):
        # Test mixed FP32 + BF16/FP16
        for grad_sample_mode in ["hooks", "ghost", "functorch"]:
            for wrap_model in [True, False]:
                for dtype in low_precision_type:
                    with self.subTest(
                        grad_sample_mode=grad_sample_mode,
                        wrap_model=wrap_model,
                        dtype=dtype,
                    ):
                        if (
                            grad_sample_mode == "functorch"
                            and model_class == EmbeddingBagModel
                        ):
                            continue
                        self._train_mixed_precision(
                            model=model_class(**model_kwargs),  # Create a fresh model
                            dataloader=dataloader,
                            dtype=dtype,
                            grad_sample_mode=grad_sample_mode,
                            wrap_model=wrap_model,
                        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv2d_model_precision(self):
        """Test mixed and low precision training with 2D convolutional layer"""
        self._test_precision_training(
            model_class=Conv2DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv3d_model_precision(self):
        """Test mixed and low precision training with 3D convolutional layer"""
        self._test_precision_training(
            model_class=Conv3DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_rnn_model_precision(self):
        """Test mixed and low precision training with RNN layers"""
        self._test_precision_training(
            model_class=RNNModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_attention_model_precision(self):
        """Test mixed and low precision training with attention layers"""
        self._test_precision_training(
            model_class=AttentionModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_heads": 4,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_complex_model_precision(self):
        """Test mixed and low precision training with a complex model combining multiple layer types"""
        self._test_precision_training(
            model_class=ComplexModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "seq_len": self.seq_len,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_conv1d_model_precision(self):
        """Test mixed precision training with 1D convolutional layer"""
        self._test_precision_training(
            model_class=Conv1DModel,
            model_kwargs={
                "input_channels": 3,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_linear_model_precision(self):
        """Test mixed and low precision training with a simple linear model"""
        self._test_precision_training(
            model_class=SimpleLinearModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_embedding_model_precision(self):
        """Test mixed and low precision training with embedding layer"""
        self._test_precision_training(
            model_class=EmbeddingModel,
            model_kwargs={
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_embedding_bag_model_precision(self):
        """Test mixed and low precision training with embedding bag layer"""
        self._test_precision_training(
            model_class=EmbeddingBagModel,
            model_kwargs={
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available, skipping test")
    def test_linear_relu_model_precision(self):
        """Test mixed and low precision training with a simple linear-relu model"""
        self._test_precision_training(
            model_class=SimpleLinearReluModel,
            model_kwargs={
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        )


if __name__ == "__main__":
    unittest.main()
