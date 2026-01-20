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
from typing import List

import torch.nn as nn
from opacus.utils.module_utils import clone_module, get_submodule, trainable_modules
from opacus.validators.errors import (
    IllegalModuleConfigurationError,
    UnsupportedModuleError,
)


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
    def fix(cls, module: nn.Module, **kwargs) -> nn.Module:
        """
        Make the module and sub_modules DP compatible by running registered custom fixers.

        Args:
            module: The root module to be made compatible.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Fixed module.
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
