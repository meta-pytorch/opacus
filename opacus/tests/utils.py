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

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicSupportedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=2)
        self.gn = nn.GroupNorm(num_groups=2, num_channels=8)
        self.fc = nn.Linear(in_features=4, out_features=8)
        self.ln = nn.LayerNorm([8, 8])

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.fc(x)
        x = self.ln(x)
        return x


class CustomLinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self._weight = nn.Parameter(torch.randn(out_features, in_features))
        self._bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return F.linear(x, self._weight, self._bias)


class MatmulModule(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_features, output_features))

    def forward(self, x):
        return torch.matmul(x, self.weight)


class LinearWithExtraParam(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(in_features, hidden_dim)
        self.extra_param = nn.Parameter(torch.randn(hidden_dim, out_features))

    def forward(self, x):
        x = self.fc(x)
        x = x.matmul(self.extra_param)
        return x
