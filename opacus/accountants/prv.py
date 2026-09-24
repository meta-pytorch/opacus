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

from typing import List

import numpy as np

from .accountant import IAccountant
from .analysis.prv import (
    Domain,
    PoissonSubsampledGaussianPRV,
    TruncatedPrivacyRandomVariable,
    compose_heterogeneous,
    compute_safe_domain_size,
    discretize,
)


class PRVAccountant(IAccountant):
    r"""
    Tracks privacy expenditure via numerical composition of Privacy loss Random
    Variables (PRVs) using the approach suggested by Gopi et al[1]. The implementation
    here is heavily inspired by the implementation of the authors that accompanied
    their paper[2].

    By utilising the Fast Fourier transform, this accountant is able to efficiently
    calculate tight bounds on the privacy expenditure, and has been shown
    experimentally to obtain tighter bounds than the RDP accountant.

    The idea behind this accountant is approximately as follows:

    A differentially private mechanism can be characterised by a PRV. The composition
    of multiple differentially privacy mechanisms can be charaterised by the sum of the
    corresponding PRVs. To get the density of the sum of PRVs, we convolve the
    individual densities.

    This accountant discretizes the PRVs corresponding to each step of the
    optimization, and convolves the approximations using the Fast Fourier Transform.
    The mesh size and bounds for the discretization are chosen automatically to ensure
    suitable approximation quality.

    The resulting convolved density is used to recover epsilon. For more detail, see
    the paper[1].

    References:
        [1] https://arxiv.org/abs/2106.02848
        [2] https://github.com/microsoft/prv_accountant
    """

    def __init__(self):
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            (last_noise_multiplier, last_sample_rate, num_steps) = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(
        self,
        delta: float,
        *,
        eps_error: float = 0.01,
        delta_error: float = None,
        **kwargs,
    ) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            eps_error: acceptable level of error in the epsilon estimate
            delta_error: acceptable level of error in delta
        """
        if delta_error is None:
            delta_error = delta / 1000
        # we construct a discrete PRV from the history
        dprv = self._get_dprv(eps_error=eps_error, delta_error=delta_error)
        # this discrete PRV can be used to directly estimate and bound epsilon
        _, _, eps_upper = dprv.compute_epsilon(delta, delta_error, eps_error)
        # return upper bound as we want guarantee, not just estimate
        return max(eps_upper, 0.0)

    def _get_dprv(self, eps_error, delta_error):
        # convert history to privacy loss random variables (prvs). Opacus currently
        # operates under the assumption that only a Poisson-subsampled Gaussian
        # mechanism is ever used during optimisation
        prvs = [
            PoissonSubsampledGaussianPRV(sample_rate, noise_multiplier)
            for noise_multiplier, sample_rate, _ in self.history
        ]
        # compute a safe domain for discretization per Gopi et al. This determines both
        # the mesh size and the truncation upper and lower bounds.
        num_self_compositions = [steps for _, _, steps in self.history]
        domain = self._get_domain(
            prvs=prvs,
            num_self_compositions=num_self_compositions,
            eps_error=eps_error,
            delta_error=delta_error,
        )
        tprvs = [
            TruncatedPrivacyRandomVariable(prv, domain.t_min, domain.t_max)
            for prv in prvs
        ]
        # discretize and convolve prvs
        dprvs = [discretize(tprv, domain) for tprv in tprvs]
        return compose_heterogeneous(
            dprvs=dprvs, num_self_compositions=num_self_compositions
        )

    def _get_domain(
        self,
        prvs: List[PoissonSubsampledGaussianPRV],
        num_self_compositions: List[int],
        eps_error: float,
        delta_error: float,
    ) -> Domain:
        total_self_compositions = sum(num_self_compositions)

        L = compute_safe_domain_size(
            prvs=prvs,
            max_self_compositions=num_self_compositions,
            eps_error=eps_error,
            delta_error=delta_error,
        )

        mesh_size = eps_error / np.sqrt(
            total_self_compositions * np.log(12 / delta_error) / 2
        )

        return Domain.create_aligned(-L, L, mesh_size)

    @classmethod
    def mechanism(cls) -> str:
        return "prv"
