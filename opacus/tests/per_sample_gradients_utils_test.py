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
from typing import Callable

import hypothesis.strategies as st
import torch
from hypothesis import given, settings
from opacus.utils.per_sample_gradients_utils import (
    check_per_sample_gradients_are_correct,
    get_grad_sample_modes,
    get_per_sample_gradient_diagnostics,
)
from torch import nn

from .grad_samples.common import expander, shrinker


class PerSampleGradientsUtilsTest(unittest.TestCase):
    def per_sample_grads_utils_test(
        self,
        x,
        model,
        grad_sample_mode,
        is_empty=False,
        atol=10e-5,
        rtol=10e-4,
    ):
        if is_empty:
            with self.assertRaises(RuntimeError):
                check_per_sample_gradients_are_correct(
                    x,
                    model,
                    batch_first=True,
                    atol=atol,
                    rtol=rtol,
                    grad_sample_mode=grad_sample_mode,
                )
            return

        assert check_per_sample_gradients_are_correct(
            x,
            model,
            batch_first=True,
            atol=atol,
            rtol=rtol,
            grad_sample_mode=grad_sample_mode,
        )

    @given(
        N=st.integers(0, 4),
        C=st.sampled_from([1, 3, 32]),
        W=st.integers(6, 10),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.integers(2, 3),
        stride=st.integers(1, 2),
        padding=st.sampled_from([0, 1, 2, "same", "valid"]),
        dilation=st.integers(1, 2),
        groups=st.integers(1, 12),
        grad_sample_mode=st.sampled_from(get_grad_sample_modes(use_ew=True)),
    )
    @settings(deadline=60000)
    def test_conv1d(
        self,
        N: int,
        C: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
        grad_sample_mode: str,
    ):
        if padding == "same" and stride != 1:
            return
        out_channels = out_channels_mapper(C)
        if (
            C % groups != 0 or out_channels % groups != 0
        ):  # since in_channels and out_channels must be divisible by groups
            return

        x = torch.randn([N, C, W])
        conv = nn.Conv1d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        ew_compatible = N > 0

        if not ew_compatible and grad_sample_mode == "ew":
            return

        self.per_sample_grads_utils_test(x, conv, grad_sample_mode, N == 0)

    @given(
        N=st.integers(0, 4),
        Z=st.integers(1, 4),
        H=st.integers(1, 3),
        W=st.integers(10, 17),
        input_dim=st.integers(2, 4),
        bias=st.booleans(),
        batch_first=st.booleans(),
        grad_sample_mode=st.sampled_from(get_grad_sample_modes(use_ew=True)),
    )
    @settings(deadline=60000)
    def test_linear(
        self,
        N: int,
        Z: int,
        H: int,
        W: int,
        input_dim: int,
        bias: bool,
        batch_first: bool,
        grad_sample_mode: str,
    ):
        if input_dim == 2:
            if not batch_first:
                return  # see https://github.com/pytorch/opacus/pull/265
            else:
                x_shape = [N, W]
        if input_dim == 3:
            x_shape = [N, Z, W]
        if input_dim == 4:
            x_shape = [N, Z, H, W]

        linear = nn.Linear(W, W + 2, bias=bias)
        x = torch.randn(x_shape)
        if not batch_first:
            x = x.transpose(0, 1)
        ew_compatible = N > 0

        if not ew_compatible and grad_sample_mode == "ew":
            return

        self.per_sample_grads_utils_test(x, linear, grad_sample_mode, N == 0)


class DiagnosticsUtilsTest(unittest.TestCase):
    """Tests for the public get_per_sample_gradient_diagnostics API."""

    def test_diagnostics_output_structure(self):
        """Verify the diagnostics report has the expected dictionary structure."""
        model = nn.Linear(8, 4)
        x = torch.randn(2, 8)

        report = get_per_sample_gradient_diagnostics(x, model)

        # Top-level keys
        self.assertIn("passed", report)
        self.assertIn("num_parameters", report)
        self.assertIn("reductions", report)
        self.assertIsInstance(report["passed"], bool)
        self.assertIsInstance(report["num_parameters"], int)

        # Reduction-level keys
        for reduction in ["sum", "mean"]:
            self.assertIn(reduction, report["reductions"])
            red_report = report["reductions"][reduction]
            self.assertIn("passed", red_report)
            self.assertIn("parameters", red_report)

            # Parameter-level keys
            for param_name, param_report in red_report["parameters"].items():
                self.assertIn("passed", param_report)
                self.assertIn("shape_match", param_report)
                self.assertIn("opacus_shape", param_report)
                self.assertIn("microbatch_shape", param_report)
                self.assertIn("opacus_l2_norm", param_report)
                self.assertIn("microbatch_l2_norm", param_report)
                self.assertIn("mse", param_report)
                self.assertIn("l1_loss", param_report)

    def test_diagnostics_linear_passes(self):
        """Verify that a standard nn.Linear model passes diagnostics."""
        model = nn.Linear(10, 5, bias=True)
        x = torch.randn(4, 10)

        report = get_per_sample_gradient_diagnostics(x, model)

        self.assertTrue(report["passed"])
        self.assertEqual(report["num_parameters"], 2)  # weight + bias
        for reduction in ["sum", "mean"]:
            red_report = report["reductions"][reduction]
            self.assertTrue(red_report["passed"])
            for param_name, param_report in red_report["parameters"].items():
                self.assertTrue(param_report["passed"])
                self.assertTrue(param_report["shape_match"])

    def test_layer_norm(self):
        """Verify diagnostics pass for nn.LayerNorm."""
        W = 8
        model = nn.LayerNorm(W, elementwise_affine=True)
        x = torch.randn(4, 6, W)

        report = get_per_sample_gradient_diagnostics(x, model)
        self.assertTrue(report["passed"])

    def test_public_import_path(self):
        """Verify the public import from opacus.utils works."""
        from opacus.utils import (
            check_per_sample_gradients_are_correct as check_fn,
        )
        from opacus.utils import (
            get_per_sample_gradient_diagnostics as diag_fn,
        )

        self.assertTrue(callable(check_fn))
        self.assertTrue(callable(diag_fn))
