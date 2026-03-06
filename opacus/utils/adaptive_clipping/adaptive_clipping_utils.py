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
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample.grad_sample_module_fast_gradient_clipping import (
    GradSampleModuleFastGradientClipping,
)
from opacus.optimizers import DPOptimizerFastGradientClipping
from opacus.privacy_engine import PrivacyEngine
from opacus.utils.fast_gradient_clipping_utils import (
    DPLossFastGradientClipping,
    DPTensorFastGradientClipping,
)
from torch.nn.parallel import DistributedDataParallel as DDP


class DPTensorFastGradientAdaptiveClipping(DPTensorFastGradientClipping):
    """
    Packages the training loop for Adaptive clipping (with Fast Gradient and Ghost Clipping) into loss.backward().
    Differently from DPTensorFastGradientClipping, the clipping norm is updated with each backward pass.
    The clipping norm track a quantile of the per-sample gradient norms.
    """

    def __init__(
        self,
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        loss_per_sample: torch.Tensor,
        loss_reduction: str = "mean",
        target_unclipped_quantile: float = 0.5,
        min_clipbound: float = 1,
        max_clipbound: float = 1e8,
        clipbound_learning_rate: float = 0.2,
        initial_noise_multiplier: float = 1.0,
    ):
        """

        Args:
            module: the module to train
            optimizer: the optimizer used to train the module
            loss_per_sample: loss on each sample in the mini-batch of size [batch_size, 1]
            target_unclipped_quantile: target quantile for unclipped gradients, between 0 and 1
            min_clipbound: minimum clipping norm allowed
            max_clipbound: maximum clipping norm allowed
            clipbound_learning_rate: learning rate for the descent algorithm that finds the target unclipped quantile
            initial_noise_multiplier: initial noise multiplier provided at step 0

        """

        super().__init__(module, optimizer, loss_per_sample, loss_reduction)

        self.target_unclipped_quantile = target_unclipped_quantile
        self.min_clipbound = min_clipbound
        self.max_clipbound = max_clipbound
        self.clipbound_learning_rate = clipbound_learning_rate
        self.initial_clipping_norm = self.optimizer.max_grad_norm
        self.initial_noise_multiplier = initial_noise_multiplier

    def backward(self):
        """
        Repurposes loss.backward() to perform two backward passes, as well as the loss rescaling and hook operations in between.
        In addition, the clipping norm is updated between the two backward passes according to a quantile of the per-sample gradient norms.
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

        # calc per_sample gradient norms
        per_sample_norms = self.module.get_norm_sample()

        # calculate new max grad norm and noise multiplier
        new_max_grad_norm, new_noise_multiplier = self._update_clip_and_noise(
            per_sample_norms
        )

        # update max grad norm and adjusted noise multiplier for noise generation
        # Note: We set _adjusted_noise_multiplier instead of noise_multiplier to preserve
        # the original noise_multiplier for correct privacy accounting (see Theorem 1 in
        # https://arxiv.org/pdf/1905.03871.pdf)
        self.module.max_grad_norm = new_max_grad_norm
        self.optimizer.max_grad_norm = new_max_grad_norm
        self.optimizer._adjusted_noise_multiplier = new_noise_multiplier

        # get the loss rescaling coefficients using the updated max_grad_norm
        coeff = torch.where(
            per_sample_norms <= self.module.max_grad_norm,
            torch.ones_like(per_sample_norms),
            self.module.max_grad_norm / per_sample_norms,
        )  # per-sample coeff, shape = [batch_size]

        second_loss_per_sample = coeff * self.loss_per_sample
        second_loss = torch.sum(second_loss_per_sample)
        self.module.disable_hooks()
        second_loss.backward()
        self.module.enable_hooks()

    def _is_distributed(self):

        return isinstance(self.module, (DPDDP, DDP))

    def _update_clip_and_noise(self, per_sample_norms):

        assert (
            self.module.max_grad_norm == self.optimizer.max_grad_norm
        ), "Max grad norm does not match between optimizer and model."

        # calculate new max_grad_norm
        current_max_norm = self.module.max_grad_norm
        local_batch_size = len(per_sample_norms)
        local_unclipped_num = (per_sample_norms <= current_max_norm).sum()

        if self._is_distributed():
            # pair the two variables in one tensor to perform only one all_reduce call
            global_unclipped_and_batch = torch.tensor(
                [local_unclipped_num, local_batch_size]
            )
            torch.distributed.all_reduce(
                global_unclipped_and_batch, op=torch.distributed.ReduceOp.SUM
            )
            unclipped_num = global_unclipped_and_batch[0].item()
            batch_size = global_unclipped_and_batch[1].item()
        else:
            unclipped_num = local_unclipped_num
            batch_size = local_batch_size

        unclipped_num_std = (
            batch_size / 20.0
        )  # use heuristic from [ATMR'22, https://arxiv.org/pdf/1905.03871]
        unclipped_num = (
            unclipped_num
            + torch.normal(mean=0.0, std=unclipped_num_std, size=(1,)).item()
        )
        unclipped_frac = unclipped_num / batch_size

        new_max_grad_norm = current_max_norm * torch.exp(
            -self.clipbound_learning_rate
            * (unclipped_frac - self.target_unclipped_quantile)
        )
        new_max_grad_norm = new_max_grad_norm.clamp(
            min=self.min_clipbound, max=self.max_clipbound
        ).item()

        # the following ensures that the updated noise multiplier is a real number
        assert (
            batch_size > 10 * self.initial_noise_multiplier
        ), "Batch size is too small. For adaptive clipping, please use a batch size larger than 10 * noise_multiplier."
        if self.initial_noise_multiplier > 0:
            # From Theorem 1 in [ATMR'22, https://arxiv.org/pdf/1905.03871]
            # The factor of 2.0 comes from the recentering of a binary bit
            # Privacy definition is for add/remove DP
            # Note: For uniform batches, the batch size is public
            # For Poisson batches, the batch size is private, and the computation below leaks some privacy, as it assumes a known batch size.
            # We currently ignore the privacy leak due to the the private batch size for Poisson subsampling of batches
            new_noise_multiplier = (
                self.initial_noise_multiplier ** (-2)
                - (2.0 * unclipped_num_std) ** (-2)
            ) ** (-1 / 2.0)
        else:
            new_noise_multiplier = self.initial_noise_multiplier

        return new_max_grad_norm, new_noise_multiplier


class DPLossFastGradientAdaptiveClipping(DPLossFastGradientClipping):
    """
    Wrapper on the loss function to be used with Adaptive Clipping (together with Fast Gradient and Ghost Clipping).
    It computes the per-sample loss, and wraps it in DPTensorFastGradientAdaptiveClipping.
    """

    def __init__(
        self,
        module: GradSampleModuleFastGradientClipping,
        optimizer: DPOptimizerFastGradientClipping,
        criterion,
        loss_reduction: str = "mean",
        target_unclipped_quantile: float = 0.5,
        min_clipbound: float = 1,
        max_clipbound: float = 1e8,
        clipbound_learning_rate: float = 0.2,
        initial_noise_multiplier: float = 1.0,
    ):

        super().__init__(module, optimizer, criterion, loss_reduction)

        self.target_unclipped_quantile = target_unclipped_quantile
        self.min_clipbound = min_clipbound
        self.max_clipbound = max_clipbound
        self.clipbound_learning_rate = clipbound_learning_rate
        self.initial_noise_multiplier = initial_noise_multiplier

    def __call__(self, *args, **kwargs) -> DPTensorFastGradientAdaptiveClipping:
        """
        Redefining the forward function to compute per-sample loss and wrap it in DPTensorFastGradientAdaptiveClipping
        """

        loss_per_sample = self.criterion(
            *args,
            **kwargs,
        )
        return DPTensorFastGradientAdaptiveClipping(
            self.module,
            self.optimizer,
            loss_per_sample,
            self.loss_reduction,
            self.target_unclipped_quantile,
            self.min_clipbound,
            self.max_clipbound,
            self.clipbound_learning_rate,
            self.initial_noise_multiplier,
        )


class PrivacyEngineAdaptiveClipping(PrivacyEngine):

    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False):
        super().__init__(accountant=accountant, secure_mode=secure_mode)

    def _prepare_criterion(
        self,
        *,
        module: GradSampleModule,
        optimizer: DPOptimizerFastGradientClipping,
        criterion=torch.nn.CrossEntropyLoss(),
        loss_reduction: str = "mean",
        target_unclipped_quantile: float = 0.5,
        min_clipbound: float = 1,
        max_clipbound: float = 1e8,
        clipbound_learning_rate: float = 0.2,
        **kwargs,
    ) -> DPLossFastGradientAdaptiveClipping:
        """
        Args:
            module: the module to train
            optimizer: the optimizer used to train the module
            criterion: the loss function used to train the module
            loss_reduction: "mean" or "sum", indicates if the loss reduction (for aggregating the gradients)
            target_unclipped_quantile: target quantile for unclipped gradients, between 0 and 1
            min_clipbound: minimum clipping norm allowed
            max_clipbound: maximum clipping norm allowed
            clipbound_learning_rate: learning rate for the descent algorithm that finds the target unclipped quantile
        """

        return DPLossFastGradientAdaptiveClipping(
            module,
            optimizer,
            criterion,
            loss_reduction=loss_reduction,
            target_unclipped_quantile=target_unclipped_quantile,
            min_clipbound=min_clipbound,
            max_clipbound=max_clipbound,
            clipbound_learning_rate=clipbound_learning_rate,
            initial_noise_multiplier=optimizer.noise_multiplier,
        )
