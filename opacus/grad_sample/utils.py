# !/usr/bin/env python3
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

from typing import Sequence, Type, Union

import torch.nn as nn

from .grad_sample_module import GradSampleHooks, GradSampleModule
from .grad_sample_module_fast_gradient_clipping import (
    GradSampleHooksFastGradientClipping,
    GradSampleModuleFastGradientClipping,
)
from .grad_sample_module_fast_gradient_clipping_fsdp import (
    GradSampleHooksFastGradientClippingFSDP,
    GradSampleModuleFastGradientClippingFSDP,
)
from .grad_sample_module_fast_gradient_clipping_tp import (
    GradSampleHooksFastGradientClippingTP,
    GradSampleModuleFastGradientClippingTP,
)
from .gsm_base import AbstractGradSampleModule
from .gsm_exp_weights import GradSampleModuleExpandedWeights
from .gsm_no_op import GradSampleHooksNoOp, GradSampleModuleNoOp


def register_grad_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
):
    """
    Registers the decorated function as the ``grad_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient
    of ``target_class_or_classes``. The signature of every grad_sampler is always the same:

    >>> @register_grad_sampler(MyCustomModel)
    ... def compute_grad_sample(module, activations, backprops):
    ...    pass

    It may help you to take a look at the existing grad_samplers inside Opacus, under ``opacus.grad_sample.``
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleHooks.GRAD_SAMPLERS[target_class] = f
            GradSampleHooksFastGradientClipping.GRAD_SAMPLERS[target_class] = f
        return f

    return decorator


def register_norm_sampler(
    target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]],
):
    """
    Registers the decorated function as the ``norm_sampler`` of ``target_class_or_classes``, which is
    the function that will be invoked every time you want to compute a per-sample gradient norm
    of ``target_class_or_classes``. The signature of every norm_sampler is always the same:

    >>> @register_norm_sampler(MyCustomModel)
    ... def compute_grad_norm_sample(module, activations, backprops):
    ...    pass
    """

    def decorator(f):
        target_classes = (
            target_class_or_classes
            if isinstance(target_class_or_classes, Sequence)
            else [target_class_or_classes]
        )
        for target_class in target_classes:
            GradSampleHooksFastGradientClipping.NORM_SAMPLERS[target_class] = f
        return f

    return decorator


def get_gsm_class(grad_sample_mode: str) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass corresponding to the input mode.

    This is used for the wrapping approach where the model is wrapped in a
    GradSampleModule subclass.

    See README for detailed comparison between grad sample modes.

    Args:
        grad_sample_mode: Mode for computing per-sample gradients. Supported values:
            - "hooks": Standard hook-based computation (GradSampleModule)
            - "functorch": Functorch-based computation (GradSampleModule with force_functorch=True)
            - "ew": Expanded weights approach (GradSampleModuleExpandedWeights)
            - "ghost": Ghost clipping with wrapping (GradSampleModuleFastGradientClipping)
            - "ghost_fsdp": Ghost clipping with FSDP (GradSampleModuleFastGradientClippingFSDP)
            - "ghost_tp": Ghost clipping with TP (GradSampleModuleFastGradientClippingTP)
            - "no_op": No-op implementation (GradSampleModuleNoOp)

    Returns:
        AbstractGradSampleModule subclass

    Raises:
        ValueError: If grad_sample_mode is not recognized
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    elif grad_sample_mode == "ghost":
        return GradSampleModuleFastGradientClipping
    elif grad_sample_mode == "ghost_fsdp":
        return GradSampleModuleFastGradientClippingFSDP
    elif grad_sample_mode == "ghost_tp":
        return GradSampleModuleFastGradientClippingTP
    elif grad_sample_mode == "no_op":
        return GradSampleModuleNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, functorch, ew, ghost, ghost_fsdp, ghost_tp, no_op"
        )


def get_hooks_class(grad_sample_mode: str):
    """
    Returns Hooks subclass corresponding to the input mode.

    This is used for the approach where hooks are attached
    directly to the model without wrapping.

    See README for a detailed comparison between grad sample modes.

    Args:
        grad_sample_mode: Mode for computing per-sample gradients. Supported values:
            - "hooks": Standard hook-based computation (GradSampleHooks)
            - "functorch": Functorch-based computation (GradSampleHooks with force_functorch=True)
            - "ghost": Ghost clipping without wrapping (GradSampleHooksFastGradientClipping)
            - "ghost_fsdp": Ghost clipping with FSDP (GradSampleHooksFastGradientClippingFSDP)
            - "ghost_tp": Ghost clipping with TP (GradSampleHooksFastGradientClippingTP)
            - "no_op": No-op implementation (GradSampleHooksNoOp)

    Returns:
        Hooks subclass

    Raises:
        ValueError: If grad_sample_mode is not recognized or not supported
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        return GradSampleHooks
    elif grad_sample_mode == "ghost":
        return GradSampleHooksFastGradientClipping
    elif grad_sample_mode == "ghost_fsdp":
        return GradSampleHooksFastGradientClippingFSDP
    elif grad_sample_mode == "ghost_tp":
        return GradSampleHooksFastGradientClippingTP
    elif grad_sample_mode == "no_op":
        return GradSampleHooksNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Hooks-based approach supports: hooks, functorch, ghost, ghost_fsdp, ghost_tp, no_op"
        )


def wrap_model(
    model: nn.Module,
    grad_sample_mode: str,
    wrap_model: bool = True,
    *args,
    **kwargs,
):
    """
    Wraps a model for per-sample gradient computation.

    This is a unified interface that supports both wrapping-based and hooks-based
    approaches for computing per-sample gradients.

    Args:
        model: PyTorch module to be wrapped or controlled
        grad_sample_mode: Mode for computing per-sample gradients
        wrap_model: If True (default), wraps model in GradSampleModule subclass.
            If False, uses hooks-based approach (no wrapping).
        *args: Additional positional arguments passed to the wrapper/hooks
        **kwargs: Additional keyword arguments passed to the wrapper/hooks

    Returns:
        Either:
        - GradSampleModule subclass instance (if wrap_model=True)
        - Hooks instance (if wrap_model=False)

    Notes:
        - When wrap_model=False, the original model is NOT wrapped and can be used
          as-is. The hooks are managed on the side.
        - When wrap_model=True, the model is wrapped and should be used via the
          returned wrapper object.
    """
    # Set force_functorch flag for functorch mode
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True

    if wrap_model:
        cls = get_gsm_class(grad_sample_mode)
    else:
        cls = get_hooks_class(grad_sample_mode)

    return cls(model, *args, **kwargs)
