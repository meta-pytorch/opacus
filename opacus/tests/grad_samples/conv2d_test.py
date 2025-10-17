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

from typing import Callable

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.grad_sample.conv import convolution2d_backward_as_a_convolution
from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.utils.tensor_utils import unfold2d
from torch.testing import assert_close

from .common import GradSampleHooks_test, expander, shrinker


class Conv2d_test(GradSampleHooks_test):
    @given(
        N=st.integers(0, 4),
        C=st.sampled_from([1, 3, 32]),
        H=st.integers(11, 17),
        W=st.integers(11, 17),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.integers(2, 4),
        stride=st.integers(1, 2),
        padding=st.sampled_from([0, 2, "same", "valid"]),
        dilation=st.integers(1, 3),
        groups=st.integers(1, 16),
    )
    @settings(deadline=60000)
    def test_conv2d(
        self,
        N: int,
        C: int,
        H: int,
        W: int,
        out_channels_mapper: Callable[[int], int],
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        groups: int,
    ):
        if padding == "same" and stride != 1:
            return
        out_channels = out_channels_mapper(C)
        if (
            C % groups != 0 or out_channels % groups != 0
        ):  # since in_channels and out_channels must be divisible by groups
            return

        x = torch.randn([N, C, H, W])
        conv = nn.Conv2d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        is_ew_compatible = (
            padding != "same" and N > 0
        )  # TODO add support for padding = 'same' with EW

        # Test regular GSM
        self.run_test(
            x,
            conv,
            batch_first=True,
            atol=10e-5,
            rtol=10e-4,
            ew_compatible=is_ew_compatible,
        )

        if padding != "same" and N > 0:
            # Test 'convolution as a backward' GSM
            # 'convolution as a backward' doesn't support padding=same
            conv2d_gsm = GradSampleModule.GRAD_SAMPLERS[nn.Conv2d]
            GradSampleModule.GRAD_SAMPLERS[nn.Conv2d] = (
                convolution2d_backward_as_a_convolution
            )
            self.run_test(
                x,
                conv,
                batch_first=True,
                atol=10e-5,
                rtol=10e-4,
                ew_compatible=is_ew_compatible,
            )
            GradSampleModule.GRAD_SAMPLERS[nn.Conv2d] = conv2d_gsm

    @given(
        B=st.integers(1, 4),
        C=st.sampled_from([1, 3, 32]),
        H=st.integers(11, 17),
        W=st.integers(11, 17),
        k_h=st.integers(2, 3),
        k_w=st.integers(2, 3),
        stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_h=st.sampled_from([0, 2]),
        pad_w=st.sampled_from([0, 2]),
        dilation_h=st.integers(1, 3),
        dilation_w=st.integers(1, 3),
    )
    @settings(deadline=60000)
    def test_unfold2d(
        self,
        B: int,
        C: int,
        H: int,
        W: int,
        k_h: int,
        k_w: int,
        pad_h: int,
        pad_w: int,
        stride_h: int,
        stride_w: int,
        dilation_h: int,
        dilation_w: int,
    ):
        X = torch.randn(B, C, H, W)
        X_unfold_torch = torch.nn.functional.unfold(
            X,
            kernel_size=(k_h, k_w),
            padding=(pad_h, pad_w),
            stride=(stride_w, stride_h),
            dilation=(dilation_w, dilation_h),
        )

        X_unfold_opacus = unfold2d(
            X,
            kernel_size=(k_h, k_w),
            padding=(pad_h, pad_w),
            stride=(stride_w, stride_h),
            dilation=(dilation_w, dilation_h),
        )

        assert_close(X_unfold_torch, X_unfold_opacus, atol=0, rtol=0)

    def test_asymetric_dilation_and_kernel_size(self) -> None:
        """
        This test is mainly for particular use cases and can be useful for future debugging
        """
        x = torch.randn(1, 1, 8, 1)
        layer = nn.Conv2d(
            1, 1, kernel_size=(3, 1), groups=1, dilation=(3, 1), bias=None
        )

        m = GradSampleModule(layer)

        y = m(x)
        backprops = torch.randn(*y.shape)
        y.backward(backprops)

        self.assertLess(
            torch.norm(m._module.weight.grad_sample[0] - m._module.weight.grad), 2e-7
        )
