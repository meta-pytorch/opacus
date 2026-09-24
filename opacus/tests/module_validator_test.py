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
from collections import OrderedDict

import torch
import torch.nn as nn
from opacus.validators.errors import UnsupportedModuleError
from opacus.validators.module_validator import ModuleValidator
from torchvision.models import mobilenet_v3_small


class ModuleValidator_test(unittest.TestCase):
    def setUp(self):
        self.original_model = mobilenet_v3_small()
        self.fixed_model = ModuleValidator.fix(self.original_model)

    def test_is_valid(self):
        self.assertFalse(ModuleValidator.is_valid(self.original_model))
        self.assertTrue(ModuleValidator.is_valid(self.fixed_model))

    def test_validate_invalid_model(self):
        with self.assertRaises(UnsupportedModuleError):
            ModuleValidator.validate(self.original_model, strict=True)
        errors = ModuleValidator.validate(self.original_model)
        self.assertGreater(len(errors), 0)

    def test_validate_valid_model(self):
        errors = ModuleValidator.validate(self.fixed_model)
        self.assertEqual(len(errors), 0)

    def test_validate_training_mode(self):
        self.fixed_model.eval()
        self.assertFalse(ModuleValidator.is_valid(self.fixed_model))
        self.fixed_model.train()
        self.assertTrue(ModuleValidator.is_valid(self.fixed_model))

    def test_is_valid_unsupported_grdsample_module(self):
        unsupported_module = nn.Bilinear(2, 2, 2)  # currently not implemented
        # ModelValidator no longer checks for supported modules
        self.assertTrue(ModuleValidator.is_valid(unsupported_module))

    def test_is_valid_extra_param(self):
        class SampleNetWithExtraParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 16)
                self.extra_param = nn.Parameter(torch.Tensor(16, 2))

            def forward(self, x):
                x = self.fc(x)
                x = x.matmul(self.extra_param)
                return x

        model = SampleNetWithExtraParam()
        # ModelValidator no longer checks for supported modules
        self.assertTrue(ModuleValidator.is_valid(model))

        model.extra_param.requires_grad = False
        self.assertTrue(ModuleValidator.is_valid(model))

    def test_fix(self):
        with self.assertLogs(level="INFO") as log_cm:
            if torch.cuda.is_available():
                self.original_model.to(torch.device("cuda:0"))
            ModuleValidator.fix(self.original_model)
            self.assertGreater(len(log_cm.records), 0)
            for log_record in log_cm.records:
                log_msg = log_record.getMessage()
                self.assertRegex(
                    log_msg,
                    "Replaced sub_module .+ with .*"
                    "|"
                    "The default batch_norm fixer replaces BatchNorm with GroupNorm.",
                )

    def test_fix_device(self):
        orig_model_device = next(self.original_model.parameters()).device
        if torch.cuda.is_available():
            self.original_model.to(torch.device("cuda:0"))
        fixed_model = ModuleValidator.fix(self.original_model)
        if torch.cuda.is_available():
            self.assertTrue(next(fixed_model.parameters()).is_cuda)
        self.original_model.to(orig_model_device)

    def test_fix_w_replace_bn_with_in(self):
        all_modules_before = self.original_model.modules()
        self.assertTrue(
            all(
                [
                    not isinstance(module, nn.InstanceNorm2d)
                    for module in all_modules_before
                ]
            )
        )

        fixed_model = ModuleValidator.fix(self.original_model, replace_bn_with_in=True)
        all_modules_after = fixed_model.modules()
        self.assertTrue(
            any([isinstance(module, nn.InstanceNorm2d) for module in all_modules_after])
        )

    def test_is_valid_non_learnable_bn(self):
        class SampleNetWithNonLearnableBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = nn.Conv2d(8, 3, 4, 4)
                self.b1 = nn.BatchNorm2d(3)
                self.c2 = nn.Conv2d(8, 3, 4, 4)
                self.b2 = nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                x = self.c1(x)
                x = self.b1(x)
                x = self.c2(x)
                x = self.b2(x)
                return x

        model = SampleNetWithNonLearnableBN()
        self.assertFalse(ModuleValidator.is_valid(model))

        model.b1.weight.requires_grad = False
        model.b1.bias.requires_grad = False
        self.assertTrue(ModuleValidator.is_valid(model))

    def test_fix_bn_with_args(self) -> None:
        m = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(4, 8)),
                    ("bn", nn.BatchNorm2d(16)),
                ]
            )
        )

        m1 = ModuleValidator.fix(m)
        self.assertTrue(isinstance(m1.bn, nn.GroupNorm))
        self.assertEqual(m1.bn.num_groups, 16)

        m2 = ModuleValidator.fix(m, num_groups=4)
        self.assertTrue(isinstance(m2.bn, nn.GroupNorm))
        self.assertEqual(m2.bn.num_groups, 4)

        m3 = ModuleValidator.fix(m, replace_bn_with_in=True)
        self.assertTrue(isinstance(m3.bn, nn.InstanceNorm2d))

        with self.assertRaises(ValueError):
            ModuleValidator.fix(m, replace_bn_with_in=True, num_groups=4)

    def test_validate_error_includes_module_name(self):
        """Verify that validation errors include the module path and type."""
        model = nn.Sequential(
            OrderedDict(
                [
                    ("fc", nn.Linear(4, 8)),
                    ("bn", nn.BatchNorm1d(8)),
                ]
            )
        )
        errors = ModuleValidator.validate(model)
        self.assertGreater(len(errors), 0)
        # The error message should contain the module name "bn" and type "BatchNorm1d"
        error_message = str(errors[0])
        self.assertIn("bn", error_message)
        self.assertIn("BatchNorm1d", error_message)

    def test_validate_error_includes_nested_module_name(self):
        """Verify that validation errors include the full nested module path."""
        model = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("linear", nn.Linear(4, 8)),
                                    ("norm", nn.BatchNorm1d(8)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )
        errors = ModuleValidator.validate(model)
        self.assertGreater(len(errors), 0)
        # The error message should contain the full path "block.norm"
        error_message = str(errors[0])
        self.assertIn("block.norm", error_message)
        self.assertIn("BatchNorm1d", error_message)
