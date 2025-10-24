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

from .conv import compute_conv_grad_sample  # noqa
from .dp_multihead_attention import compute_sequence_bias_grad_sample  # noqa
from .dp_rnn import compute_rnn_linear_grad_sample  # noqa
from .embedding import compute_embedding_grad_sample  # noqa
from .embedding_norm_sample import compute_embedding_norm_sample  # noqa
from .grad_sample_controller import GradSampleController  # noqa
from .grad_sample_module import GradSampleModule, create_or_accumulate_grad_sample
from .grad_sample_module_fast_gradient_clipping import (  # noqa
    GradSampleModuleFastGradientClipping,
)
from .grad_sample_module_fast_gradient_clipping_fsdp import (  # noqa
    GradSampleModuleFastGradientClippingFSDP,
)
from .grad_sample_module_fast_gradient_clipping_tp import (  # noqa
    GradSampleModuleFastGradientClippingTP,
)
from .group_norm import compute_group_norm_grad_sample  # noqa
from .gsm_base import AbstractGradSampleModule
from .gsm_exp_weights import GradSampleModuleExpandedWeights
from .gsm_no_op import GradSampleModuleNoOp
from .instance_norm import compute_instance_norm_grad_sample  # noqa
from .layer_norm import compute_layer_norm_grad_sample  # noqa
from .linear import compute_linear_grad_sample  # noqa
from .rms_norm import compute_rms_norm_grad_sample  # noqa
from .utils import (
    get_gsm_class,
    register_grad_sampler,
    register_norm_sampler,
    wrap_model,
)


__all__ = [
    "GradSampleController",
    "GradSampleModule",
    "GradSampleModuleFastGradientClipping",
    "GradSampleModuleFastGradientClippingFSDP",
    "GradSampleModuleFastGradientClippingTP",
    "GradSampleModuleExpandedWeights",
    "GradSampleModuleNoOp",
    "AbstractGradSampleModule",
    "register_grad_sampler",
    "register_norm_sampler",
    "create_or_accumulate_grad_sample",
    "wrap_model",
    "get_gsm_class",
]
