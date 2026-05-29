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

from typing import Optional

import hypothesis.strategies as st
import torch
import torch.nn as nn
from hypothesis import given, settings
from opacus.layers import DPMultiheadAttention

from .common import DPModules_test


def remove_in_proj_bias_hook(self, state_dict, prefix, local_metadata):
    return {
        param_name: param_value
        for param_name, param_value in state_dict.items()
        if param_name != "in_proj_bias"
    }


def attn_train_fn(
    model: nn.Module,
    *args,
    **kwargs,
):
    model.train()
    criterion = nn.MSELoss()
    logits, attn_weights = model(*args, **kwargs)
    y = torch.zeros_like(logits)
    loss = criterion(logits, y)
    loss.backward()


class DPMultiheadAttention_test(DPModules_test):
    @given(
        batch_size=st.integers(1, 5),
        src_seq_len=st.integers(1, 6),
        tgt_seq_len=st.integers(1, 6),
        num_heads=st.integers(1, 3),
        bias=st.booleans(),
        add_bias_kv=st.booleans(),
        add_zero_attn=st.booleans(),
        kdim=st.integers(2, 8) | st.none(),
        vdim=st.integers(2, 8) | st.none(),
    )
    @settings(deadline=60000)
    def test_attn(
        self,
        batch_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        num_heads: int,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Optional[int],
        vdim: Optional[int],
    ):
        embed_dim = 4 * num_heads  # embed_dim must be divisible by num_heads

        attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )
        dp_attn = DPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

        dp_attn.load_state_dict(attn.state_dict())

        q = torch.randn(tgt_seq_len, batch_size, embed_dim)
        k = torch.randn(
            src_seq_len, batch_size, kdim if kdim is not None else embed_dim
        )
        v = torch.randn(
            src_seq_len, batch_size, vdim if vdim is not None else embed_dim
        )

        self.compare_forward_outputs(
            attn,
            dp_attn,
            q,
            k,
            v,
            output_names=("attn_out", "attn_out_weights"),
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

        self.compare_gradients(
            attn,
            dp_attn,
            attn_train_fn,
            q,
            k,
            v,
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

    @given(
        batch_size=st.integers(1, 5),
        src_seq_len=st.integers(1, 6),
        tgt_seq_len=st.integers(1, 6),
        num_heads=st.integers(1, 3),
        bias=st.booleans(),
        add_bias_kv=st.booleans(),
        add_zero_attn=st.booleans(),
        kdim=st.integers(2, 8) | st.none(),
        vdim=st.integers(2, 8) | st.none(),
    )
    @settings(deadline=60000)
    def test_dp_attn(
        self,
        batch_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        num_heads: int,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Optional[int],
        vdim: Optional[int],
    ):
        embed_dim = 4 * num_heads  # embed_dim must be divisible by num_heads

        attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )
        dp_attn = DPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

        attn.load_state_dict(dp_attn.state_dict())

        q = torch.randn(tgt_seq_len, batch_size, embed_dim)
        k = torch.randn(
            src_seq_len, batch_size, kdim if kdim is not None else embed_dim
        )
        v = torch.randn(
            src_seq_len, batch_size, vdim if vdim is not None else embed_dim
        )

        self.compare_forward_outputs(
            attn,
            dp_attn,
            q,
            k,
            v,
            output_names=("attn_out", "attn_out_weights"),
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

        self.compare_gradients(
            attn,
            dp_attn,
            attn_train_fn,
            q,
            k,
            v,
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
        )

    @given(
        batch_size=st.integers(1, 5),
        src_seq_len=st.integers(1, 6),
        tgt_seq_len=st.integers(1, 6),
        num_heads=st.integers(1, 3),
        bias=st.booleans(),
        batch_first=st.booleans(),
        attn_mask_dim=st.integers(2, 3),
    )
    @settings(deadline=60000)
    def test_attn_mask_batch_first(
        self,
        batch_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        num_heads: int,
        bias: bool,
        batch_first: bool,
        attn_mask_dim: int,
    ):
        embed_dim = 4 * num_heads

        attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=bias,
            batch_first=batch_first,
        )
        dp_attn = DPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=bias,
            batch_first=batch_first,
        )

        dp_attn.load_state_dict(attn.state_dict())

        if batch_first:
            q = torch.randn(batch_size, tgt_seq_len, embed_dim)
            k = torch.randn(batch_size, src_seq_len, embed_dim)
            v = torch.randn(batch_size, src_seq_len, embed_dim)
        else:
            q = torch.randn(tgt_seq_len, batch_size, embed_dim)
            k = torch.randn(src_seq_len, batch_size, embed_dim)
            v = torch.randn(src_seq_len, batch_size, embed_dim)

        if attn_mask_dim == 2:
            attn_mask = torch.zeros(tgt_seq_len, src_seq_len, dtype=torch.bool)
        else:
            attn_mask = torch.zeros(
                batch_size * num_heads, tgt_seq_len, src_seq_len, dtype=torch.bool
            )

        self.compare_forward_outputs(
            attn,
            dp_attn,
            q,
            k,
            v,
            output_names=("attn_out", "attn_out_weights"),
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=attn_mask,
        )

        self.compare_gradients(
            attn,
            dp_attn,
            attn_train_fn,
            q,
            k,
            v,
            atol=1e-5,
            rtol=1e-3,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=attn_mask,
        )

    @given(
        batch_size=st.integers(1, 5),
        src_seq_len=st.integers(1, 6),
        tgt_seq_len=st.integers(1, 6),
        num_heads=st.integers(1, 3),
        bias=st.booleans(),
        add_bias_kv=st.booleans(),
        add_zero_attn=st.booleans(),
        kdim=st.integers(2, 8) | st.none(),
        vdim=st.integers(2, 8) | st.none(),
    )
    @settings(deadline=60000)
    def test_dp_attn_hook(
        self,
        batch_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        num_heads: int,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Optional[int],
        vdim: Optional[int],
    ):
        embed_dim = 4 * num_heads  # embed_dim must be divisible by num_heads

        dp_attn = DPMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,  # Untestable between two different implementations
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
        )

        # Test if the hook function is working within the state_dict
        if "in_proj_bias" in dp_attn.state_dict():
            dp_attn._register_state_dict_hook(remove_in_proj_bias_hook)
            self.assertFalse("in_proj_bias" in dp_attn.state_dict())
