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

import io
from typing import Callable, Dict, Iterable, List, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.grad_sample import wrap_model
from opacus.utils.module_utils import trainable_parameters
from opacus.utils.packed_sequences import compute_seq_lengths
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def clone_module(module: nn.Module) -> nn.Module:
    """
    Handy utility to clone an nn.Module. PyTorch doesn't always support copy.deepcopy(), so it is
    just easier to serialize the model to a BytesIO and read it from there.

    Args:
        module: The module to clone

    Returns:
        The clone of ``module``
    """
    with io.BytesIO() as bytesio:
        torch.save(module, bytesio)
        bytesio.seek(0)
        module_copy = torch.load(bytesio, weights_only=False)
    return module_copy


def is_batch_empty(batch: Union[torch.Tensor, Iterable[torch.Tensor]]):
    if type(batch) is torch.Tensor:
        return batch.numel() == 0
    else:
        return batch[0].numel() == 0


class ModelWithLoss(nn.Module):
    """
    To test the gradients of a module, we need to have a loss.
    This module makes it easy to get a loss from any nn.Module, and automatically generates
    a target y vector for it in the forward (of all zeros of the correct size).
    This reduces boilerplate while testing.
    """

    supported_reductions = ["mean", "sum"]

    def __init__(self, module: nn.Module, loss_reduction: str = "mean"):
        """
        Instantiates this module.

        Args:
            module: The nn.Module you want to test.
            loss_reduction: What reduction to apply to the loss. Defaults to "mean".

        Raises:
            ValueError: If ``loss_reduction`` is not among those supported.
        """
        super().__init__()
        self.wrapped_module = module

        if loss_reduction not in self.supported_reductions:
            raise ValueError(
                f"Passed loss_reduction={loss_reduction}. Only {self.supported_reductions} supported."
            )
        self.criterion = nn.L1Loss(reduction=loss_reduction)

    def forward(self, x):
        if type(x) is tuple:
            x = self.wrapped_module(*x)
        else:
            x = self.wrapped_module(x)
        if type(x) is PackedSequence:
            loss = _compute_loss_packedsequences(self.criterion, x)
        else:
            y = torch.zeros_like(x)
            loss = self.criterion(x, y)
        return loss


def compute_microbatch_grad_sample(
    x: Union[torch.Tensor, List[torch.Tensor]],
    module: nn.Module,
    batch_first: bool = True,
    loss_reduction: str = "mean",
    chunk_method: Callable = iter,
) -> Dict[str, torch.tensor]:
    """
    Computes per-sample gradients with the microbatch method, i.e. by computing normal gradients
    with batch_size set to 1, and manually accumulating them. This is our reference for testing
    as this method is obviously correct, but slow.

    Args:
        x: Sample input batch
        module: The nn.Module you want to test.
        batch_first: Whether batch size is the first dimension (as opposed to the second).
            Defaults to True.
        loss_reduction: What reduction to apply to the loss. Defaults to "mean".
        chunk_method: The method to use to split the batch into microbatches. Defaults to ``iter``.

    Returns:
        Dictionary mapping parameter_name -> per-sample-gradient for that parameter
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)

    module = ModelWithLoss(clone_module(module), loss_reduction)

    for _, p in trainable_parameters(module):
        p.microbatch_grad_sample = []

    if not batch_first and type(x) is not list:
        # This allows us to iterate with x_i
        x = x.transpose(0, 1)

    # Invariant: x is [B, T, ...]

    for x_i in chunk_method(x):
        # x_i is [T, ...]
        module.zero_grad()
        if type(x_i) is not tuple:
            # EmbeddingBag provides tuples
            x_i = x_i.unsqueeze(
                0 if batch_first else 1
            )  # x_i of size [1, T, ...] if batch_first, else [T, 1, ...]
        loss_i = module(x_i)
        loss_i.backward()
        for p in module.parameters():
            p.microbatch_grad_sample.append(p.grad.detach().clone())

    for _, p in trainable_parameters(module):
        if batch_first:
            p.microbatch_grad_sample = torch.stack(
                p.microbatch_grad_sample, dim=0  # [B, T, ...]
            )
        else:
            p.microbatch_grad_sample = torch.stack(
                p.microbatch_grad_sample, dim=1  # [T, B, ...]
            ).transpose(
                0, 1
            )  # Opacus's semantics is that grad_samples are ALWAYS batch_first: [B, T, ...]

    microbatch_grad_samples = {
        name: p.microbatch_grad_sample
        for name, p in trainable_parameters(module.wrapped_module)
    }
    return microbatch_grad_samples


def compute_opacus_grad_sample(
    x: Union[torch.Tensor, PackedSequence],
    module: nn.Module,
    batch_first: bool = True,
    loss_reduction: str = "mean",
    grad_sample_mode: str = "hooks",
) -> Dict[str, torch.tensor]:
    """
    Runs Opacus to compute per-sample gradients and return them for testing purposes.

    Args:
        x: Sample input batch
        module: The nn.Module you want to test.
        batch_first: Whether batch size is the first dimension (as opposed to the second).
            Defaults to True.
        loss_reduction: What reduction to apply to the loss. Defaults to "mean".
        grad_sample_mode: What sampling method to use to get gradients.

    Returns:
        Dictionary mapping parameter_name -> per-sample-gradient for that parameter
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)

    gs_module = wrap_model(
        model=clone_module(module),
        grad_sample_mode=grad_sample_mode,
        batch_first=batch_first,
        loss_reduction=loss_reduction,
    )
    grad_sample_module = ModelWithLoss(gs_module, loss_reduction)

    grad_sample_module.zero_grad()
    loss = grad_sample_module(x)
    loss.backward()

    opacus_grad_samples = {
        name: p.grad_sample
        for name, p in trainable_parameters(grad_sample_module.wrapped_module._module)
    }

    return opacus_grad_samples


def check_torch_version_for_ew_sample() -> bool:
    return torch.__version__ >= (1, 13)


def get_grad_sample_modes(use_ew: bool = False):
    grad_sample_modes = ["hooks", "functorch"]
    if use_ew and check_torch_version_for_ew_sample():
        grad_sample_modes.append("ew")
    return grad_sample_modes


def check_per_sample_gradients_are_correct(
    x: Union[torch.Tensor, PackedSequence],
    module: nn.Module,
    *,
    batch_first: bool = True,
    atol: float = 10e-6,
    rtol: float = 10e-5,
    grad_sample_mode: str = "hooks",
) -> bool:
    """
    A utility to check whether per sample gradients are computed correctly with a particular model.
    The check is performed by comparing the result of the slow but reliable micro-batch method `compute_microbatch_grad_sample`
    with the result of optimized opacus method.

    Args:
        x: Sample input batch
        module: The ``ModelWithLoss`` that wraps the nn.Module you want to check.
        batch_first: Whether batch size is the first dimension (as opposed to the second).
            Defaults to True.
        atol: The relative tolerance parameter (torch.allclose).
        rtol: The absolute tolerance parameter (torch.allclose).
        grad_sample_mode: What sampling method to use to get gradients.

    Returns: True if per sample gradients were computed correctly. False otherwise.

    Example:
        >>> N, Z, W = 100, 10, 10
        >>> x_shape = [N, Z, W]
        >>> x = torch.randn(x_shape)
        >>> model = nn.Linear(W, W + 2)
        >>> assert check_per_sample_gradients_are_correct(
        ...            x,
        ...            model
        ...        ) # This will fail only if the opacus per sample gradients do not match the micro-batch gradients.
    """
    reductions = ["sum", "mean"]
    if grad_sample_mode == "ew":
        if not batch_first:
            raise RuntimeError("Batch should be first dimension.")
        if not check_torch_version_for_ew_sample():
            raise RuntimeError(f"Unsupported torch version: {torch.__version__}.")

    for loss_reduction in reductions:
        if not _check_per_sample_gradients_are_correct_with_reduction(
            x,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            atol=atol,
            rtol=rtol,
            grad_sample_mode=grad_sample_mode,
        ):
            return False

    return True


def compute_microbatch_grad_sample_tensor_or_seq(
    x: Union[torch.Tensor, PackedSequence],
    module: nn.Module,
    batch_first: bool = True,
    loss_reduction: str = "mean",
):
    if type(x) is PackedSequence:
        x_unpacked = unpack_packedsequences(x)
        microbatch_grad_samples = compute_microbatch_grad_sample(
            x_unpacked,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
        )
    else:
        microbatch_grad_samples = compute_microbatch_grad_sample(
            x, module, batch_first=batch_first, loss_reduction=loss_reduction
        )

    return microbatch_grad_samples


def compute_grad_samples_microbatch_and_opacus(
    x: Union[torch.Tensor, PackedSequence],
    module: nn.Module,
    batch_first: bool = True,
    loss_reduction: str = "mean",
    grad_sample_mode: str = "hooks",
    chunk_method: Callable = iter,
):
    if type(x) is PackedSequence:
        x_unpacked = unpack_packedsequences(x)
        microbatch_grad_samples = compute_microbatch_grad_sample(
            x_unpacked,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            chunk_method=chunk_method,
        )
    elif not is_batch_empty(x):
        microbatch_grad_samples = compute_microbatch_grad_sample(
            x,
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            chunk_method=chunk_method,
        )
    else:
        raise RuntimeError("x is expected to be non-empty.")

    opacus_grad_samples = compute_opacus_grad_sample(
        x,
        module,
        batch_first=batch_first,
        loss_reduction=loss_reduction,
        grad_sample_mode=grad_sample_mode,
    )

    if microbatch_grad_samples.keys() != opacus_grad_samples.keys():
        raise ValueError(
            "Keys not matching! "
            f"Keys only in microbatch: {microbatch_grad_samples.keys() - opacus_grad_samples.keys()}; "
            f"Keys only in Opacus: {opacus_grad_samples.keys() - microbatch_grad_samples.keys()}"
        )

    return microbatch_grad_samples, opacus_grad_samples


def _check_per_sample_gradients_are_correct_with_reduction(
    x: Union[torch.Tensor, PackedSequence],
    module: nn.Module,
    batch_first: bool = True,
    loss_reduction: str = "mean",
    atol: float = 10e-6,
    rtol: float = 10e-5,
    grad_sample_mode: str = "hooks",
) -> bool:
    (
        microbatch_grad_samples,
        opacus_grad_samples,
    ) = compute_grad_samples_microbatch_and_opacus(
        x,
        module,
        batch_first=batch_first,
        loss_reduction=loss_reduction,
        grad_sample_mode=grad_sample_mode,
    )

    for name, opacus_grad_sample in opacus_grad_samples.items():
        microbatch_grad_sample = microbatch_grad_samples[name]
        if not opacus_grad_sample.shape == microbatch_grad_sample.shape:
            return False
        if not torch.allclose(microbatch_grad_sample, opacus_grad_sample, atol, rtol):
            return False
    return True


def unpack_packedsequences(X: PackedSequence) -> List[torch.Tensor]:
    r"""
    Produces a list of tensors from X (PackedSequence) such that this list was used to create X with batch_first=True

    Args:
        X: A PackedSequence from which the output list of tensors will be produced.

    Returns:
        unpacked_data: The list of tensors produced from X.
    """

    X_padded = pad_packed_sequence(X)
    X_padded = X_padded[0].permute((1, 0, 2))

    if X.sorted_indices is not None:
        X_padded = X_padded[X.sorted_indices]

    seq_lens = compute_seq_lengths(X.batch_sizes)
    unpacked_data = [0] * len(seq_lens)
    for idx, length in enumerate(seq_lens):
        unpacked_data[idx] = X_padded[idx][:length, :]

    return unpacked_data


def _compute_loss_packedsequences(
    criterion: nn.L1Loss, x: PackedSequence
) -> torch.Tensor:
    r"""
    This function computes the loss in a different way for 'mean' reduced L1 loss while for 'sum' reduced L1 loss,
    it computes the same way as with non-packed data. For 'mean' reduced L1 loss, it transforms x (PackedSequence)
    into a list of tensors such that this list of tensors was used to create this PackedSequence in the first
    place using batch_first=True and then takes the mean of the loss values produced from applying criterion on
    each sequence sample.

    Args:
        criterion: An L1 loss function with reduction either set to 'sum' or 'mean'.
        x: Data in the form of a PackedSequence.

    Returns:
        A loss variable, reduced either using summation or averaging from L1 errors.
    """

    if criterion.reduction == "sum":
        y = torch.zeros_like(x[0])
        return criterion(x[0], y)
    elif criterion.reduction == "mean":
        x = unpack_packedsequences(x)
        loss_sum = 0
        for x_i in x:
            y_i = torch.zeros_like(x_i)
            loss_sum += criterion(x_i, y_i)
        loss_mean = loss_sum / len(x)
        return loss_mean
