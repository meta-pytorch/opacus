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

import torch
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizerFastGradientClipping


class DPTensorFastGradientClipping:
    """
    Packages the training loop for Fast Gradient and Ghost Clipping into loss.backward().
    """

    def __init__(
        self,
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        loss_per_sample: torch.Tensor,
        loss_reduction: str = "mean",
    ):
        """
        Args:
            module: the module to train
            optimizer: the optimizer used to train the module
            loss_per_sample: loss on each sample in the mini-batch of size [batch_size, 1]
        """
        self.module = module
        self.optimizer = optimizer
        self.loss_per_sample = loss_per_sample
        self.loss_reduction = loss_reduction

    def item(self):
        return self.detach().item()

    def detach(self):
        if self.loss_reduction == "mean":
            return torch.mean(self.loss_per_sample).detach()
        elif self.loss_reduction == "sum":
            return torch.sum(self.loss_per_sample).detach()

    def __truediv__(self, other):
        """
        Division operation for DPTensorFastGradientClipping.
        Enables: loss / scalar
        """
        return DPTensorFastGradientClipping(
            self.module,
            self.optimizer,
            self.loss_per_sample / other,
            self.loss_reduction,
        )

    def __mul__(self, other):
        """
        Multiplication operation for DPTensorFastGradientClipping.
        Enables: loss * scalar or scalar * loss
        """
        return DPTensorFastGradientClipping(
            self.module,
            self.optimizer,
            self.loss_per_sample * other,
            self.loss_reduction,
        )

    def __rmul__(self, other):
        """
        Left multiplication by a scalar.
        Required to support loss weighting: weight * loss
        """
        return self.__mul__(other)

    def __add__(self, other):
        """
        Addition operation for DPTensorFastGradientClipping.
        Enables: loss + scalar or loss + loss.
        Required to support combining multiple losses in a single training step.
        """
        if isinstance(other, DPTensorFastGradientClipping):
            if self.loss_reduction != other.loss_reduction:
                raise ValueError(
                    f"Cannot add losses with different reductions: {self.loss_reduction} vs {other.loss_reduction}"
                )
            return DPTensorFastGradientClipping(
                self.module,
                self.optimizer,
                self.loss_per_sample + other.loss_per_sample,
                self.loss_reduction,
            )
        else:
            return DPTensorFastGradientClipping(
                self.module,
                self.optimizer,
                self.loss_per_sample + other,
                self.loss_reduction,
            )

    def __radd__(self, other):
        """
        Right addition operation for DPTensorFastGradientClipping.
        Enables: scalar + loss
        """
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction operation for DPTensorFastGradientClipping.
        Enables: loss - scalar or loss - loss
        """
        if isinstance(other, DPTensorFastGradientClipping):
            if self.loss_reduction != other.loss_reduction:
                raise ValueError(
                    f"Cannot subtract losses with different reductions: {self.loss_reduction} vs {other.loss_reduction}"
                )
            return DPTensorFastGradientClipping(
                self.module,
                self.optimizer,
                self.loss_per_sample - other.loss_per_sample,
                self.loss_reduction,
            )
        else:
            return DPTensorFastGradientClipping(
                self.module,
                self.optimizer,
                self.loss_per_sample - other,
                self.loss_reduction,
            )

    def __rsub__(self, other):
        """
        Right subtraction operation for DPTensorFastGradientClipping.
        Enables: scalar - loss
        """
        return DPTensorFastGradientClipping(
            self.module,
            self.optimizer,
            other - self.loss_per_sample,
            self.loss_reduction,
        )

    def __neg__(self):
        """
        Negation operation for DPTensorFastGradientClipping.
        Enables: -loss
        """
        return DPTensorFastGradientClipping(
            self.module,
            self.optimizer,
            -self.loss_per_sample,
            self.loss_reduction,
        )

    def __repr__(self):
        """String representation"""
        return f"DPTensorFastGradientClipping(loss_reduction={self.loss_reduction}, shape={self.loss_per_sample.shape})"

    def __str__(self):
        """String representation"""
        return f"DPTensorFastGradientClipping({self.item():.4f})"

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
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        criterion,
        loss_reduction: str = "mean",
    ):
        assert loss_reduction in [
            "mean",
            "sum",
        ], "loss_reduction should be either 'mean' or 'sum'"

        # if the criterion is missing reduction attribute, use module's reduction attribute'
        if not hasattr(criterion, "reduction"):
            setattr(criterion, "reduction", module.loss_reduction)

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
        old_reduction = self.criterion.reduction
        self.criterion.reduction = "none"
        loss_per_sample = self.criterion(*args, **kwargs)
        self.criterion.reduction = old_reduction

        if shape is not None and loss_per_sample.shape[0] == shape[0] * shape[1]:
            # Note that the privacy unit for generative NLP tasks is per sequence.
            # The shape variable is the shape of the logits before flattening i.e., [batch_size, sequence_lenght, vocab_size].
            # This variable is necessary for ghost clipping to work with generative NLP tasks.
            loss_per_sample = loss_per_sample.view(shape[0], shape[1])  # BxT
            if self.loss_reduction == "mean":
                # When the criterion has ignore_index, positions matching it
                # produce zero loss but should also be excluded from the
                # denominator (matching PyTorch's CrossEntropyLoss behavior).
                ignore_index = getattr(self.criterion, "ignore_index", None)
                if ignore_index is not None and len(args) >= 2:
                    targets = args[1]
                    if "target" in kwargs:
                        targets = kwargs["target"]
                    mask = targets.view(shape[0], shape[1]) != ignore_index
                    num_valid = mask.sum(dim=1).clamp(min=1)
                    loss_per_sample = loss_per_sample.sum(dim=1) / num_valid  # B
                else:
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
