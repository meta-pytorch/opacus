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

from .common import GradSampleHooks_test, expander, shrinker


class Conv1d_test(GradSampleHooks_test):
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
        self.run_test(
            x, conv, batch_first=True, atol=10e-5, rtol=10e-4, ew_compatible=N > 0
        )

    def test_conv1d_padding_modes(self):
        # Non-zero padding modes pad the input before the convolution
        for padding_mode in ["reflect", "replicate", "circular"]:
            for padding in [2, "same"]:
                with self.subTest(padding_mode=padding_mode, padding=padding):
                    x = torch.randn([4, 4, 8])
                    conv = nn.Conv1d(
                        in_channels=4,
                        out_channels=6,
                        kernel_size=3,
                        padding=padding,
                        dilation=2,
                        groups=2,
                        padding_mode=padding_mode,
                    )
                    self.run_test(
                        x,
                        conv,
                        batch_first=True,
                        atol=10e-5,
                        rtol=10e-4,
                        ew_compatible=padding != "same",
                    )
