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

# we import fix and validate methods from all submodules here
# to ensure that when `opacus.validators` is imported,
# we call register_module_validator and register_module_fixer
# on respective methods

from .batch_norm import fix, validate  # noqa
from .gru import fix, validate  # noqa
from .instance_norm import fix, validate  # noqa
from .lstm import fix, validate  # noqa
from .module_validator import ModuleValidator
from .multihead_attention import fix, validate  # noqa
from .utils import register_module_fixer, register_module_validator


__all__ = [
    "ModuleValidator",
    "register_module_validator",
    "register_module_fixer",
]
