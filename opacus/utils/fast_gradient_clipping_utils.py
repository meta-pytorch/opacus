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

from typing import Union

import torch
from opacus.grad_sample import (
    GradSampleHooksFastGradientClipping,
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizerFastGradientClipping


class DPTensorFastGradientClipping:
    """
    Packages the training loop for Fast Gradient and Ghost Clipping into loss.backward().
    """

    def __init__(
        self,
        module: Union[
            GradSampleModuleFastGradientClipping, GradSampleHooksFastGradientClipping
        ],
        optimizer: DPOptimizerFastGradientClipping,
        loss_per_sample: torch.Tensor,
        loss_reduction: str = "mean",
    ):
        """

        Args:
            module: the (GradSample) module to train or hooks handling object
            optimizer: the optimizer used to train the module
            loss_per_sample: loss on each sample in the mini-batch of size [batch_size, 1]

        """

        self.module = module
        self.optimizer = optimizer
        self.loss_per_sample = loss_per_sample
        self.loss_reduction = loss_reduction

    def item(self):
        if self.loss_reduction == "mean":
            return torch.mean(self.loss_per_sample).detach().item()
        elif self.loss_reduction == "sum":
            return torch.sum(self.loss_per_sample).detach().item()

    def backward(self):
        """
        Repurposes loss.backward() to perform two backward passes, as well as the loss rescaling and hook operations in between
        """

        if self.loss_reduction == "mean":
            reduced_loss = torch.mean(self.loss_per_sample, dim=0)
        elif self.loss_reduction == "sum":
            reduced_loss = torch.sum(self.loss_per_sample, dim=0)
        else:
            raise ValueError(
                f"loss_reduction = {self.loss_reduction}. Only 'sum' and 'mean' losses are supported"
            )
        reduced_loss.backward(retain_graph=True)
        self.optimizer.zero_grad()
        coeff = self.module.get_clipping_coef()
        second_loss_per_sample = (
            coeff.to(self.loss_per_sample.device) * self.loss_per_sample
        )
        second_loss = torch.sum(second_loss_per_sample)
        self.module.disable_hooks()
        second_loss.backward()
        self.module.enable_hooks()


class DPLossFastGradientClipping:
    """
    Wrapper on the loss function to be used with Fast Gradient and Ghost Clipping. It computes the per-sample loss, and wraps it in DPTensorFastGradientClipping.
    """

    def __init__(
        self,
        module: Union[
            GradSampleModuleFastGradientClipping, GradSampleHooksFastGradientClipping
        ],
        optimizer: DPOptimizerFastGradientClipping,
        criterion,
        loss_reduction: str = "mean",
    ):
        assert loss_reduction in [
            "mean",
            "sum",
        ], "loss_reduction should be either 'mean' or 'sum'"

        assert (
            loss_reduction
            == criterion.reduction
            == module.loss_reduction
            == optimizer.loss_reduction
        ), "loss_reduction should be the same across GradSampleModule, Optimizer, Criterion, and loss_reduction"

        self.optimizer = optimizer
        self.module = module
        self.criterion = criterion
        self.loss_reduction = loss_reduction

    def __call__(self, *args, shape=None, **kwargs) -> DPTensorFastGradientClipping:
        """
        Redefining the forward function to compute per-sample loss and wrap it in DPTensorFastGradientClipping
        """

        loss_per_sample = self.criterion(*args, **kwargs)

        if shape is not None and loss_per_sample.shape[0] == shape[0] * shape[1]:
            # Note that the privacy unit for generative NLP tasks is per sequence.
            # The shape variable is the shape of the logits before flattening i.e., [batch_size, sequence_lenght, vocab_size].
            # This variable is necessary for ghost clipping to work with generative NLP tasks.
            loss_per_sample = loss_per_sample.view(shape[0], shape[1])  # BxT
            if self.loss_reduction == "mean":
                loss_per_sample = loss_per_sample.mean(dim=1)  # B
            elif self.loss_reduction == "sum":
                loss_per_sample = loss_per_sample.sum(dim=1)  # B
            else:
                raise ValueError(
                    f"loss_reduction = {self.loss_reduction}. Only 'sum' and 'mean' losses are supported"
                )

        return DPTensorFastGradientClipping(
            self.module, self.optimizer, loss_per_sample, self.loss_reduction
        )
