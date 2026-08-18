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

r"""
Implements privacy accounting for Gaussian Differential Privacy.
Applies the Dual and Central Limit Theorem (CLT) to estimate privacy budget of
an iterated subsampled Gaussian Mechanism (by either uniform or Poisson
subsampling).
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm


def compute_mu_uniform(
    *, steps: int, noise_multiplier: float, sample_rate: float
) -> float:
    """
    Compute mu from uniform subsampling.

    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Sample rate

    Returns:
        mu
    """

    c = sample_rate * np.sqrt(steps)
    return (
        np.sqrt(2)
        * c
        * np.sqrt(
            np.exp(noise_multiplier ** (-2)) * norm.cdf(1.5 / noise_multiplier)
            + 3 * norm.cdf(-0.5 / noise_multiplier)
            - 2
        )
    )


def compute_mu_poisson(
    *, steps: int, noise_multiplier: float, sample_rate: float
) -> float:
    """
    Compute mu from uniform subsampling.

    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Sample rate

    Returns:
        mu
    """

    return np.sqrt(np.exp(noise_multiplier ** (-2)) - 1) * np.sqrt(steps) * sample_rate


def delta_eps_mu(*, eps: float, mu: float) -> float:
    """
    Compute dual between mu-GDP and (epsilon, delta)-DP.

    Args:
        eps: eps
        mu: mu
    """
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def eps_from_mu(*, mu: float, delta: float) -> float:
    """
    Compute epsilon from mu given delta via inverse dual.

    Args:
        mu:
        delta:
    """

    def f(x):
        """Reversely solve dual by matching delta."""
        return delta_eps_mu(eps=x, mu=mu) - delta

    if f(0) <= 0:
        return 0.0

    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root


def compute_eps_uniform(
    *, steps: int, noise_multiplier: float, sample_rate: float, delta: float
) -> float:
    """
    Compute epsilon given delta from inverse dual of uniform subsampling.

    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Sample rate
        delta: Target delta

    Returns:
        eps
    """

    return eps_from_mu(
        mu=compute_mu_uniform(
            steps=steps, noise_multiplier=noise_multiplier, sample_rate=sample_rate
        ),
        delta=delta,
    )


def compute_eps_poisson(
    *, steps: int, noise_multiplier: float, sample_rate: float, delta: float
) -> float:
    """
    Compute epsilon given delta from inverse dual of Poisson subsampling

    Args:
        steps: Number of steps taken
        noise_multiplier: Noise multiplier (sigma)
        sample_rate: Sample rate
        delta: Target delta

    Returns:
        eps
    """

    return eps_from_mu(
        mu=compute_mu_poisson(
            steps=steps, noise_multiplier=noise_multiplier, sample_rate=sample_rate
        ),
        delta=delta,
    )
