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
import logging
import sys
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def has_trainable_params(module: nn.Module) -> bool:
    return any(p.requires_grad for p in module.parameters(recurse=False))


def parametrized_modules(module: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters (as opposed to "wrapper modules" that just organize modules).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in module.named_modules()
        if any(p is not None for p in m.parameters(recurse=False))
    )


def trainable_modules(module: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """
    Recursively iterates over all submodules, returning those that
    have parameters and are trainable (ie they want a grad).
    """
    yield from (
        (m_name, m)
        for (m_name, m) in parametrized_modules(module)
        if any(p.requires_grad for p in m.parameters(recurse=False))
    )


def trainable_parameters(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    """
    Recursively iterates over all parameters, returning those that
    are trainable (ie they want a grad).
    """
    yield from (
        (p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad
    )


def requires_grad(module: nn.Module, *, recurse: bool = False) -> bool:
    """
    Checks if any parameters in a specified module require gradients.

    Args:
        module: PyTorch module whose parameters are to be examined.
        recurse: Flag specifying if the gradient requirement check should
            be applied recursively to submodules of the specified module

    Returns:
        Flag indicating if any parameters require gradients
    """
    requires_grad = any(p.requires_grad for p in module.parameters(recurse))
    return requires_grad


def clone_module(module: nn.Module) -> nn.Module:
    """
    Handy utility to clone an nn.Module. PyTorch doesn't always support copy.deepcopy(), so it is
    just easier to serialize the model to a BytesIO and read it from there.
    When ``weights_only=False``, ``torch.load()`` uses the "pickle" module implicitly, which is known to be insecure.
    Only load the model you trust.

    Args:
        module: The module to clone

    Returns:
        The clone of ``module``
    """
    with io.BytesIO() as bytesio:
        torch.save(module, bytesio)
        bytesio.seek(0)
        module_copy = torch.load(bytesio, weights_only=False)
    next_param = next(
        module.parameters(), None
    )  # Eg, InstanceNorm with affine=False has no params
    return module_copy.to(next_param.device) if next_param is not None else module_copy


def get_submodule(module: nn.Module, target: str) -> nn.Module:
    """
    Returns the submodule given by target if it exists, otherwise throws an error.

    This is copy-pasta of Pytorch 1.9's ``get_submodule()`` implementation; and is
    included here to also support Pytorch 1.8. This function can be removed in favour
    of ``module.get_submodule()`` once Opacus abandons support for torch 1.8.

    See more details at https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=get_submodule#torch.nn.Module.get_submodule

    Args:
        module: module
        target: submodule string

    Returns:
        The submodule given by target if it exists

    Raises:
        AttributeError
            If submodule doesn't exist
    """

    if target == "":
        return module

    atoms: List[str] = target.split(".")
    mod: nn.Module = module

    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(
                mod._get_name() + " has no " "attribute `" + item + "`"
            )
        mod = getattr(mod, item)
        if not isinstance(mod, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")
    return mod


def are_state_dict_equal(sd1: Dict, sd2: Dict):
    """
    Compares two state dicts, while logging discrepancies
    """
    if len(sd1) != len(sd2):
        return False

    for k1, v1 in sd1.items():
        # check that all keys are accounted for.
        if k1 not in sd2:
            return False
        # check that value tensors are equal.
        v2 = sd2[k1]
        if not torch.allclose(v1, v2):
            return False
    return True
