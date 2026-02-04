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

import abc
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Mapping, TypeVar

from opacus.optimizers import DPOptimizer


T_state_dict = TypeVar("T_state_dict", bound=Mapping[str, Any])


class IAccountant(abc.ABC):
    @abc.abstractmethod
    def __init__(self):
        self.history = []  # history of noise multiplier, sample rate, and steps

    @abc.abstractmethod
    def step(self, *, noise_multiplier: float, sample_rate: float):
        """
        Signal one optimization step

        Args:
            noise_multiplier: Current noise multiplier
            sample_rate: Current sample rate
        """
        pass

    @abc.abstractmethod
    def get_epsilon(self, delta: float, *args, **kwargs) -> float:
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            *args: subclass-specific args
            **kwargs: subclass-specific kwargs
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Number of optimization steps taken so far
        """
        pass

    @classmethod
    @abc.abstractmethod
    def mechanism(cls) -> str:
        """
        Accounting mechanism name
        """
        pass

    def get_optimizer_hook_fn(
        self, sample_rate: float
    ) -> Callable[[DPOptimizer], None]:
        """
        Returns a callback function which can be used to attach to DPOptimizer
        Args:
            sample_rate: Expected sampling rate used for accounting
        """

        def hook_fn(optim: DPOptimizer):
            # This works for Poisson for both single-node and distributed
            # The reason is that the sample rate is the same in both cases (but in
            # distributed mode, each node samples among a subset of the data)
            self.step(
                noise_multiplier=optim.noise_multiplier,
                sample_rate=sample_rate * optim.accumulated_iterations,
            )

        return hook_fn

    def state_dict(self, destination: T_state_dict = None) -> T_state_dict:
        """
        Returns a dictionary containing the state of the accountant.
        Args:
            destination: a mappable object to populate the current state_dict into.
                If this arg is None, an OrderedDict is created and populated.
                Default: None
        """
        if destination is None:
            destination = OrderedDict()
        destination["history"] = deepcopy(self.history)
        destination["mechanism"] = self.mechanism()
        return destination

    def load_state_dict(self, state_dict: T_state_dict):
        """
        Validates the supplied state_dict and populates the current
        Privacy Accountant's state dict.

        Args:
            state_dict: state_dict to load.

        Raises:
            ValueError if supplied state_dict is invalid and cannot be loaded.
        """
        if state_dict is None or len(state_dict) == 0:
            raise ValueError(
                "state dict is either None or empty and hence cannot be loaded"
                " into Privacy Accountant."
            )
        if "history" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `history`."
                " Cannot be loaded into Privacy Accountant."
            )
        if "mechanism" not in state_dict.keys():
            raise ValueError(
                "state_dict does not have the key `mechanism`."
                " Cannot be loaded into Privacy Accountant."
            )
        if self.mechanism() != state_dict["mechanism"]:
            raise ValueError(
                f"state_dict of {state_dict['mechanism']} cannot be loaded into "
                f" Privacy Accountant with mechanism {self.mechanism()}"
            )
        self.history = state_dict["history"]
