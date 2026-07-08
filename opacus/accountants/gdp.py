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

import warnings

from .accountant import IAccountant
from .analysis import gdp as privacy_analysis


class GaussianAccountant(IAccountant):
    def __init__(self):
        warnings.warn(
            "GDP accounting is experimental and can underestimate privacy expenditure."
            "Proceed with caution. More details: https://arxiv.org/pdf/2106.02848.pdf"
        )
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier != noise_multiplier
                or last_sample_rate != sample_rate
            ):
                raise ValueError(
                    "Noise multiplier and sample rate have to stay constant in GaussianAccountant."
                )
            else:
                self.history = [
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                ]

        else:
            self.history = [(noise_multiplier, sample_rate, 1)]

    def get_epsilon(self, delta: float, poisson: bool = True, **kwargs) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            poisson: ``True`` is input batches was sampled via Poisson sampling,
                ``False`` otherwise
        """

        compute_eps = (
            privacy_analysis.compute_eps_poisson
            if poisson
            else privacy_analysis.compute_eps_uniform
        )
        noise_multiplier, sample_rate, num_steps = self.history[-1]
        return compute_eps(
            steps=num_steps,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            delta=delta,
        )

    @classmethod
    def mechanism(cls) -> str:
        return "gdp"
