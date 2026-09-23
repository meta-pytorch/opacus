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

import logging
import warnings
from typing import List

import torch.fx as fx
import torch.nn as nn
from opacus.utils.module_utils import clone_module, get_submodule, trainable_modules
from opacus.validators.errors import (
    IllegalModuleConfigurationError,
    UnsupportedModuleError,
)
from opacus.validators.inplace import disable_inplace, fix_inplace_operations


logger = logging.getLogger(__name__)


class ModuleValidator:
    """
    Encapsulates all the validation logic required by Opacus.
    Also works as a namespace to hold registered validators and fixers.
    """

    VALIDATORS = {}
    FIXERS = {}

    @classmethod
    def validate(
        cls, module: nn.Module, *, strict: bool = False
    ) -> List[UnsupportedModuleError]:
        """
        Validate module and sub_modules by running registered custom validators.
        Returns or raises exceptions depending on ``strict`` flag.

        Args:
            module: The root module to validate.
            strict: Boolean to indicate whether to raise errors or return
            the list of errors.

        Raises:
            UnsupportedModuleError in case of validation failures.
        """
        errors = []
        # 1. validate that module is in training mode
        if not module.training:
            errors.append(
                IllegalModuleConfigurationError("Model needs to be in training mode")
            )
        # 2. perform module specific validations for trainable modules.
        # TODO: use module name here - it's useful part of error message
        for _, sub_module in trainable_modules(module):
            if type(sub_module) in ModuleValidator.VALIDATORS:
                sub_module_validator = ModuleValidator.VALIDATORS[type(sub_module)]
                errors.extend(sub_module_validator(sub_module))
        # raise/return as needed
        if strict and len(errors) > 0:
            raise UnsupportedModuleError(errors)
        else:
            return errors

    @classmethod
    def is_valid(cls, module: nn.Module) -> bool:
        """
        Check if module and sub_modules are valid by running registered custom validators.

        Args:
            module: The root module to validate.

        Returns:
            bool
        """
        return len(cls.validate(module, strict=False)) == 0

    @classmethod
    def fix(
        cls, module: nn.Module, *, remove_inplace_ops: bool = False, **kwargs
    ) -> nn.Module:
        """
        Make the module and sub_modules DP compatible by running registered custom fixers.

        In addition to the per-module-type fixers, this always disables in-place
        activation flags (e.g. ``nn.ReLU(inplace=True)``) since in-place writes are
        incompatible with Opacus' backward hooks. This is a cheap, type-preserving
        fix that requires no tracing.

        Functional/tensor in-place ops baked into ``forward`` (e.g. a ResNet's
        ``out += identity``) are only rewritten when ``remove_inplace_ops=True``,
        because doing so requires symbolic tracing and changes the returned type.
        It is opt-in: models without such ops (or whose only in-place ops are the
        activation flags handled above) do not need it. If your model crashes under
        ``grad_sample_mode="hooks"`` with ``RuntimeError: Output 0 of
        BackwardHookFunction is a view and is being modified inplace``, re-run with
        ``remove_inplace_ops=True``.

        When ``remove_inplace_ops=True`` and tracing succeeds, the returned object
        is a ``torch.fx.GraphModule``, not the original ``nn.Module`` subclass.
        ``GraphModule`` is a subclass of ``nn.Module`` and preserves forward outputs
        for typical models, but ``isinstance`` checks against the original class,
        custom methods defined on the original class, pickling by class name, and
        similar type-dependent code will observe the GraphModule type instead. A
        warning is emitted in this case.

        FX tracing also records a single execution path: data-independent control
        flow in the root ``forward`` (e.g. ``if self.training``) is frozen to the
        branch taken during tracing, so a later ``.train()``/``.eval()`` toggle may
        have no effect. Standard ``nn.Dropout``/``nn.BatchNorm`` submodules keep
        their own flags and are unaffected.

        Rewriting replaces in-place writes with out-of-place equivalents. Forward
        outputs are preserved, but aliasing semantics change: a tensor mutated in
        place is instead replaced by a new tensor, so other aliases to the original
        storage will not see the update. Code relying on in-place side effects via
        aliasing may diverge in behavior.

        Models that cannot be symbolically traced (e.g. data-dependent control flow)
        are returned unchanged and the fallback is logged at INFO level.

        Args:
            module: The root module to be made compatible.
            remove_inplace_ops: If True, rewrite functional/tensor in-place ops
                out-of-place via symbolic tracing. Defaults to False. No-op if the
                module cannot be traced.
            **kwargs: Arbitrary keyword arguments forwarded to the per-module fixers.

        Returns:
            Fixed module. Type is ``torch.fx.GraphModule`` when tracing succeeds with
            ``remove_inplace_ops=True``, otherwise the original (cloned) ``nn.Module`` type.
        """
        module = clone_module(module)
        # iterate over all sub_modules
        # We have to get sub_module names in a list first as we will be
        # changing the modules inside the loop.
        sub_module_names = [name for name, _ in trainable_modules(module)]
        for sub_module_name in sub_module_names:
            # get sub_module
            sub_module = get_submodule(module, sub_module_name)
            # if sub_module has a registered fixer
            if type(sub_module) in ModuleValidator.FIXERS:
                # get a replacement for sub_module
                sub_module_fixer = ModuleValidator.FIXERS[type(sub_module)]
                new_sub_module = sub_module_fixer(sub_module, **kwargs)
                # move new_sub_module to the same device as that of sub_module
                new_sub_module.to(next(sub_module.parameters()).device)
                # get module after replacement.
                module = cls._replace_sub_module(
                    root=module,
                    sub_module_name=sub_module_name,
                    new_sub_module=new_sub_module,
                )
                # log it
                logger.info(
                    f"Replaced sub_module {sub_module_name} : {sub_module}"
                    f" with {new_sub_module}"
                )
        # in-place writes are incompatible with Opacus' backward hooks: always
        # disable in-place activation flags, and optionally rewrite functional
        # in-place ops (e.g. ``out += identity``) out-of-place.
        disable_inplace(module)
        if remove_inplace_ops:
            module = fix_inplace_operations(module)
            if isinstance(module, fx.GraphModule):
                warnings.warn(
                    "ModuleValidator.fix() traced your model with torch.fx to rewrite "
                    "any in-place operations out-of-place. Because of this, the returned "
                    "model is now a torch.fx.GraphModule instead of its original type. "
                    "Two things to be aware of:\n"
                    "  1. Anything that checks the model's type -- isinstance checks, "
                    "custom methods on your model class, or pickling -- may not behave "
                    "as before.\n"
                    "  2. If your model's forward() uses an if/else on a flag such as "
                    "self.training, tracing keeps only the branch taken while tracing, "
                    "so switching between .train() and .eval() afterwards may have no "
                    "effect. (Standard layers like Dropout and BatchNorm are not "
                    "affected.)\n"
                    "To skip tracing and keep your model's original type, call "
                    "ModuleValidator.fix(..., remove_inplace_ops=False).",
                )
        # return fixed module
        return module

    @classmethod
    def _replace_sub_module(
        cls,
        *,
        root: nn.Module,
        sub_module_name: str,
        new_sub_module: nn.Module,
    ) -> None:
        sub_module_path = sub_module_name.split(".")
        if (
            len(sub_module_path) == 1 and sub_module_path[0] == ""
        ):  # root is the only sub_module of root
            return new_sub_module
        else:  # replace root's descendant
            sub_module_parent = root
            for name in sub_module_path[:-1]:  # descend down to sub_module
                sub_module_parent = sub_module_parent._modules[name]
            sub_module_parent._modules[sub_module_path[-1]] = new_sub_module
        return root

    @classmethod
    def fix_and_validate(cls, module: nn.Module, **kwargs) -> nn.Module:
        """
        Fix the module and sub_modules first, and then run validation.

        Args:
            module: The root module to be fixed and validated
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Fixed module.

        Raises:
            UnsupportedModuleError in case of validation failures.
        """
        # 1. replace any fixable modules
        fixed_module = cls.fix(module, **kwargs)
        # 2. perform module specific validations.
        cls.validate(fixed_module, strict=True)
        # return fixed module
        return fixed_module
