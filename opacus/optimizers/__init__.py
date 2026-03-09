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

from .adaclipoptimizer import AdaClipDPOptimizer
from .ddp_perlayeroptimizer import SimpleDistributedPerLayerOptimizer
from .ddpoptimizer import DistributedDPOptimizer
from .ddpoptimizer_automatic_clipping import (
    DistributedDPAutomaticClippingOptimizer,
    DistributedDPPerLayerAutomaticClippingOptimizer,
)
from .ddpoptimizer_fast_gradient_clipping import (
    DistributedDPOptimizerFastGradientClipping,
)
from .fsdpoptimizer_fast_gradient_clipping import FSDPOptimizerFastGradientClipping
from .optimizer import DPOptimizer
from .optimizer_automatic_clipping import (
    DPAutomaticClippingOptimizer,
    DPPerLayerAutomaticClippingOptimizer,
)
from .optimizer_fast_gradient_clipping import DPOptimizerFastGradientClipping
from .perlayeroptimizer import DPPerLayerOptimizer


__all__ = [
    "AdaClipDPOptimizer",
    "DistributedDPOptimizer",
    "DPOptimizer",
    "DPOptimizerFastGradientClipping",
    "DistributedDPOptimizerFastGradientlipping",
    "FSDPOptimizerFastGradientClipping",
    "DPPerLayerOptimizer",
    "SimpleDistributedPerLayerOptimizer",
    "DPAutomaticClippingOptimizer",
    "DPPerLayerAutomaticClippingOptimizer",
    "DistributedDPAutomaticClippingOptimizer",
    "DistributedDPPerLayerAutomaticClippingOptimizer",
]


def _get_ghost_mode_optimizer(clipping: str, distributed: bool):
    """Get optimizer class for ghost grad_sample_mode."""
    if clipping != "flat":
        raise ValueError(
            f"Unsupported combination of parameters. Clipping: {clipping} and grad_sample_mode: ghost"
        )
    if distributed:
        return DistributedDPOptimizerFastGradientClipping
    return DPOptimizerFastGradientClipping


def _get_ghost_fsdp_optimizer(clipping: str, distributed: bool):
    """Get optimizer class for ghost_fsdp grad_sample_mode."""
    if clipping != "flat" or not distributed:
        raise ValueError(
            f"Unsupported combination of parameters. Clipping: {clipping}, "
            f"distributed: {distributed}, and grad_sample_mode: ghost_fsdp"
        )
    return FSDPOptimizerFastGradientClipping


def _get_per_layer_distributed_optimizer(grad_sample_mode: str):
    """Get optimizer class for per_layer distributed case."""
    if grad_sample_mode not in ("hooks", "ew"):
        raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
    return SimpleDistributedPerLayerOptimizer


def get_optimizer_class(clipping: str, distributed: bool, grad_sample_mode: str = None):
    # Handle special grad_sample_mode cases first
    if grad_sample_mode == "ghost":
        return _get_ghost_mode_optimizer(clipping, distributed)
    if grad_sample_mode == "ghost_fsdp":
        return _get_ghost_fsdp_optimizer(clipping, distributed)

    # Handle per_layer distributed case with grad_sample_mode check
    if clipping == "per_layer" and distributed:
        return _get_per_layer_distributed_optimizer(grad_sample_mode)

    # Standard lookup for common cases
    optimizer_map = {
        ("flat", False): DPOptimizer,
        ("flat", True): DistributedDPOptimizer,
        ("per_layer", False): DPPerLayerOptimizer,
        ("automatic", False): DPAutomaticClippingOptimizer,
        ("automatic", True): DistributedDPAutomaticClippingOptimizer,
        ("automatic_per_layer", False): DPPerLayerAutomaticClippingOptimizer,
        ("automatic_per_layer", True): DistributedDPPerLayerAutomaticClippingOptimizer,
        ("adaptive", False): AdaClipDPOptimizer,
    }

    key = (clipping, distributed)
    if key in optimizer_map:
        return optimizer_map[key]

    raise ValueError(
        f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
    )
