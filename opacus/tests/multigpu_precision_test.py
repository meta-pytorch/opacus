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
import torch.multiprocessing as mp
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch.nn.parallel import DistributedDataParallel as DDP

from .mixed_precision_utils import (
    EmbeddingModel,
    SimpleLinearReluModel,
    create_random_data,
)
from .multigpu_gradcheck_test import cleanup, setup


def _get_training_components(
    model_class: nn.Module,
    model_kwargs: dict[str, int],
    device: torch.device,
    grad_sample_mode: str,
    attach_only: bool = False,
):
    """
    Creates a model, optimizer, criterion, and dataloader for training.
    The model is wrapped in DPDDP or DDP.
    The optimizer, model, dataloader, and criterion are wrapped by the privacy engine.
    """
    input_dim = model_kwargs.get("input_dim", 4)
    output_dim = model_kwargs.get("output_dim", 4)
    seq_len = model_kwargs.get("seq_len", 4)
    batch_size = 2
    num_batches = 2

    dataloader, _ = create_random_data(
        model_class,
        batch_size=batch_size,
        input_dim=input_dim,
        output_dim=output_dim,
        num_batches=num_batches,
        seq_len=seq_len,
        device=device,
    )

    model = model_class(**model_kwargs).to(device)
    model = model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()

    if not attach_only:
        model = DPDDP(model)
    else:
        model = DDP(model, device_ids=[device.index])

    if grad_sample_mode in ["hooks", "functorch", "ew"]:
        hooks_or_module, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=1,
            max_grad_norm=1,
            grad_sample_mode=grad_sample_mode,
            poisson_sampling=False,
            attach_only=attach_only,
        )
    elif grad_sample_mode == "ghost":
        hooks_or_module, optimizer, criterion, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            criterion=criterion,
            max_grad_norm=1,
            noise_multiplier=1,
            grad_sample_mode="ghost",
            poisson_sampling=False,
            attach_only=attach_only,
        )

    if not attach_only:
        model = hooks_or_module

    return model, optimizer, criterion, dataloader


def run_mixed_precision_test(
    rank: int,
    world_size: int,
    model_class: nn.Module,
    model_kwargs: dict[str, int],
    dtype: torch.dtype,
    grad_sample_mode: str,
    attach_only: bool = False,
):
    """
    Runs an integration test for distributed training with DPDDP and mixed precision training.
    It check dtypes of various training artifacts.
        The expected behavior is that:
            - model parameters are in full precision FP32
            - model outputs are in low precision (BF16 or FP16)
            - gradients are in high precision (FP32)

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        model_class (nn.Module): The neural network model to be trained.
        model_kwargs (dict): The keyword arguments for the model.
        dtype (torch.dtype): The data type for low precision training (torch.float16 or torch.bfloat16).
        grad_sample_mode (str): The mode for per-sample gradient computation.
        attach_only (bool): Whether to use attach-only mode.
    """

    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model, optimizer, criterion, dataloader = _get_training_components(
        model_class, model_kwargs, device, grad_sample_mode, attach_only=attach_only
    )

    # Model weights should be in high precision (fp32)
    model = model.to(torch.float32)
    for p in model.parameters():
        assert p.dtype == torch.float32

    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", dtype=dtype):
            outputs = model(x)
            assert outputs.dtype == dtype

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()

        # The gradients should have been cast up to high precision (fp32)
        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.dtype == torch.float32
            if p.grad_sample is not None:
                if isinstance(p.grad_sample, list):
                    for gs in p.grad_sample:
                        assert gs.dtype in [torch.float32, dtype]
                else:
                    assert p.grad_sample.dtype in [torch.float32, dtype]
            if grad_sample_mode == "ghost" and p._norm_sample is not None:
                assert p._norm_sample.dtype in [torch.float32, dtype]

        optimizer.step()

    cleanup()


def run_low_precision_test(
    rank: int,
    world_size: int,
    model_class: nn.Module,
    model_kwargs: dict[str, int],
    dtype: torch.dtype,
    grad_sample_mode: str,
    attach_only: bool = False,
):
    """
    Runs an integration test for distributed training with DPDDP and low precision training.
    Tests that model weights, outputs, and gradients are in the low precision dtype.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The number of processes participating in the job.
        model_class (nn.Module): The neural network model to be trained.
        model_kwargs (dict): The keyword arguments for the model.
        dtype (torch.dtype): The data type for low precision training (torch.float16 or torch.bfloat16).
        grad_sample_mode (str): The mode for per-sample gradient computation.
        attach_only (bool): Whether to use attach-only mode.
    """
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    model, optimizer, criterion, dataloader = _get_training_components(
        model_class, model_kwargs, device, grad_sample_mode, attach_only=attach_only
    )

    # Model weights should be in low precision
    model = model.to(dtype)
    for p in model.parameters():
        assert p.dtype == dtype

    for batch in dataloader:
        x, y = batch
        optimizer.zero_grad()

        if x.is_floating_point():  # For embedding layers, keep input as int
            x = x.to(dtype)
        outputs = model(x)
        assert outputs.dtype == dtype

        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                assert p.grad.dtype == dtype
            if p.grad_sample is not None:
                if isinstance(p.grad_sample, list):
                    for gs in p.grad_sample:
                        assert gs.dtype == dtype
                else:
                    assert p.grad_sample.dtype == dtype
            if grad_sample_mode == "ghost" and p._norm_sample is not None:
                assert p._norm_sample.dtype == dtype

        optimizer.step()

    cleanup()


class MultiGPUPrecisionTest(unittest.TestCase):
    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def setUp(self):
        self.input_dim = 4
        self.hidden_dim = 16
        self.output_dim = 4
        self.seq_len = 4
        self.bf16_supported = hasattr(torch, "bfloat16")

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_precision_training(
        self,
    ):
        """
        Runs an integration test for distributed training with DPDDP and mixed and low precision training.
        Tests that model weights, outputs, and gradients are in the expected dtypes.
        Tests all available grad sample modes.
        """

        model_kwargs_map = {
            SimpleLinearReluModel: {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            EmbeddingModel: {
                "vocab_size": 100,
                "embedding_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
        }

        # test models sequentially since running tests in parallel fails for DDP
        for model_class in [SimpleLinearReluModel, EmbeddingModel]:

            model_kwargs = model_kwargs_map[model_class]
            dtype = torch.bfloat16 if self.bf16_supported else torch.float16
            world_size = 2
            # test low precision training
            for grad_sample_mode in ["ew", "functorch", "hooks", "ghost"]:
                for attach_only in [False, True]:
                    # "ew" is not supported for EmbeddingModel or attach_only=True
                    if grad_sample_mode == "ew" and (
                        model_class == EmbeddingModel or attach_only
                    ):
                        continue
                    with self.subTest(
                        model_class=model_class,
                        grad_sample_mode=grad_sample_mode,
                        attach_only=attach_only,
                        precision="low",
                    ):
                        mp.spawn(
                            run_low_precision_test,
                            args=(
                                world_size,
                                model_class,
                                model_kwargs,
                                dtype,
                                grad_sample_mode,
                                attach_only,
                            ),
                            nprocs=world_size,
                            join=True,
                        )

            # test mixed precision training
            for grad_sample_mode in ["functorch", "hooks", "ghost"]:
                for attach_only in [False, True]:
                    with self.subTest(
                        model_class=model_class,
                        grad_sample_mode=grad_sample_mode,
                        attach_only=attach_only,
                        precision="mixed",
                    ):
                        mp.spawn(
                            run_mixed_precision_test,
                            args=(
                                world_size,
                                model_class,
                                model_kwargs,
                                dtype,
                                grad_sample_mode,
                                attach_only,
                            ),
                            nprocs=world_size,
                            join=True,
                        )


if __name__ == "__main__":
    unittest.main()
