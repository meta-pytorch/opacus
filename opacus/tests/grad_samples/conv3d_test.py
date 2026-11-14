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

from typing import Tuple, Union

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings

from .common import GradSampleHooks_test, expander, shrinker


class Conv3d_test(GradSampleHooks_test):
    @given(
        N=st.integers(0, 4),
        C=st.sampled_from([1, 3, 32]),
        D=st.integers(3, 6),
        H=st.integers(6, 10),
        W=st.integers(6, 10),
        out_channels_mapper=st.sampled_from([expander, shrinker]),
        kernel_size=st.sampled_from([2, 3, (1, 2, 3)]),
        stride=st.sampled_from([1, 2, (1, 2, 3)]),
        padding=st.sampled_from([0, 2, (1, 2, 3), "same", "valid"]),
        dilation=st.sampled_from([1, (1, 2, 2)]),
        groups=st.integers(1, 16),
    )
    @settings(deadline=60000)
    def test_conv3d(
        self,
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        out_channels_mapper: int,
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]],
        padding: Union[int, Tuple[int]],
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
        x = torch.randn([N, C, D, H, W])
        conv = nn.Conv3d(
            in_channels=C,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        is_ew_compatible = (
            dilation == 1 and padding != "same" and N > 0
        )  # TODO add support for padding = 'same' with EW
        self.run_test(
            x,
            conv,
            batch_first=True,
            atol=10e-5,
            rtol=10e-3,
            ew_compatible=is_ew_compatible,
        )

    def test_conv3d_padding_modes(self):
        # Non-zero padding modes pad the input before the convolution
        for padding_mode in ["reflect", "replicate", "circular"]:
            for padding in [(1, 2, 1), "same"]:
                with self.subTest(padding_mode=padding_mode, padding=padding):
                    x = torch.randn([4, 2, 4, 5, 6])
                    conv = nn.Conv3d(
                        in_channels=2,
                        out_channels=4,
                        kernel_size=(2, 3, 3),
                        padding=padding,
                        padding_mode=padding_mode,
                    )
                    self.run_test(
                        x,
                        conv,
                        batch_first=True,
                        atol=10e-5,
                        rtol=10e-3,
                        ew_compatible=padding != "same",
                    )
