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

import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SequenceBias(nn.Module):
    r"""
    Adds one bias element to the end of the sequence.
    so if the input has a shape ``(L, N, E)``, (``batch_first = False``),
    where ``L`` is the sequence length, ``N`` is the batch size, and ``E`` is
    the embedding dimension, the output will have a shape
    ``(L+1, N, E)``. When ``batch_first = True``, input has a shape ``(N, L, E)``
    and the output will have a shape ``(N, L+1, E)``

    Attributes:
        bias (:class:`torch.nn.parameter.Parameter`): the learnable bias of
            the module of shape ``(E)``, where ``E`` is the embedding dimension.

    Example:
        >>> m = SequenceBias(16, batch_first=False)
        >>> input = torch.randn(20, 4, 16)
        >>> output = m(input)
        >>> output.size()
        torch.Size([21, 4, 16])
    """

    def __init__(self, embed_dim: int, batch_first: bool = False):
        r"""
        Args:
            embed_dim: Embedding dimension
        """
        super(SequenceBias, self).__init__()
        self.batch_first = batch_first
        self.bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        r"""
        assigns Normally distributed random values to bias.
        """
        nn.init.normal_(self.bias)

    def forward(self, x):
        if self.batch_first:
            bsz, _, _ = x.shape
            return torch.cat([x, self.bias.repeat(bsz, 1, 1)], 1)
        else:
            _, bsz, _ = x.shape
            return torch.cat([x, self.bias.repeat(1, bsz, 1)])


class DPMultiheadAttention(nn.Module):
    r"""
    This is DP-friendly implementation of nn.MultiheadAttention.
    For full reference see original module refer to
    :class:`torch.nn.MultiheadAttention`.

    Current implementation leverages pytorch modules as building blocks
    to allow DP engine to calculate per-sample gradients.
    This is in contrast with original implementation based on nn.functional.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ):
        super(DPMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # when self._qkv_same_embed_dim = True, "in_proj_weight" rather than "q,k,v_weight" and fast path calculation will be used in "nn.transformer", which should be avoided. This is why we force self._qkv_same_embed_dim = False.
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.qlinear = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.klinear = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.vlinear = nn.Linear(self.vdim, embed_dim, bias=bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.add_bias_kv = add_bias_kv
        if self.add_bias_kv:
            self.seq_bias_k = SequenceBias(embed_dim)
            self.seq_bias_v = SequenceBias(embed_dim)

        self.add_zero_attn = add_zero_attn

        self.dropout = nn.Dropout(dropout)

        # to avoid null pointers in Transformer.forward
        self.in_proj_weight = None
        self.in_proj_bias = None

    def load_state_dict(self, state_dict):
        r"""
        Loads module from previously saved state.

        Supports loading from both :class:`torch.nn.MultiheadAttention` and
        :class:`opacus.layers.dp_multihead_attention.DPMultiheadAttention`.

        Args:
            state_dict: Please refer to
                https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html.
        """
        if "in_proj_weight" in state_dict:
            qweight, kweight, vweight = state_dict["in_proj_weight"].chunk(3, dim=0)

            state_dict["qlinear.weight"] = qweight
            state_dict["klinear.weight"] = kweight
            state_dict["vlinear.weight"] = vweight
            del state_dict["in_proj_weight"]

        if "in_proj_bias" in state_dict:
            qbias, kbias, vbias = state_dict["in_proj_bias"].chunk(3, dim=0)

            state_dict["qlinear.bias"] = qbias
            state_dict["klinear.bias"] = kbias
            state_dict["vlinear.bias"] = vbias
            del state_dict["in_proj_bias"]

        if "bias_k" in state_dict:
            state_dict["seq_bias_k.bias"] = state_dict["bias_k"].squeeze()
            del state_dict["bias_k"]

        if "bias_v" in state_dict:
            state_dict["seq_bias_v.bias"] = state_dict["bias_v"].squeeze()
            del state_dict["bias_v"]

        if "q_proj_weight" in state_dict:
            state_dict["qlinear.weight"] = state_dict["q_proj_weight"]
            del state_dict["q_proj_weight"]

        if "k_proj_weight" in state_dict:
            state_dict["klinear.weight"] = state_dict["k_proj_weight"]
            del state_dict["k_proj_weight"]

        if "v_proj_weight" in state_dict:
            state_dict["vlinear.weight"] = state_dict["v_proj_weight"]
            del state_dict["v_proj_weight"]

        super(DPMultiheadAttention, self).load_state_dict(state_dict)

    # flake8: noqa C901
    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        is_causal=False,
    ):
        is_batched = query.dim() == 3

        assert is_batched == True, "The query must have a dimension of 3."

        r"""
        As per https://github.com/pytorch/opacus/issues/596, we have to include ``is_causal`` as a dummy parameter of the function,
        since it is used in the ``forward`` function of parent class ``nn.TransformerEncoderLayer``.
        """
        assert (
            is_causal == False
        ), "We currently do not support causal mask. Will fix it in the future."

        r"""
        Using the same logic with ``nn.MultiheadAttention`` (https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html).
        """

        if not self.batch_first:
            tgt_len, bsz, embed_dim = query.size()
            key_seq_len = key.size(0)
        else:
            bsz, tgt_len, embed_dim = query.size()
            key_seq_len = key.size(1)

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"query has as size of {embed_dim} while the embedding"
                " size is {self.embed_dim}"
            )

        head_dim = embed_dim // self.num_heads
        if head_dim * self.num_heads != embed_dim:
            raise ValueError(
                f"embedding dimension {embed_dim} not divisible "
                "by number of heads {num_heads}"
            )
        scaling = float(head_dim) ** -0.5

        q = self.qlinear(query)
        k = self.klinear(key)
        v = self.vlinear(value)

        q = q * scaling

        if self.batch_first:
            q, k, v = [x.transpose(0, 1) for x in (q, k, v)]

        if attn_mask is not None:
            if attn_mask.dtype not in (
                torch.float32,
                torch.float64,
                torch.uint8,
                torch.bool,
            ):
                raise ValueError(
                    f"Only float, byte, and bool types are supported for attn_mask, "
                    f"not {attn_mask.dtype}."
                )

            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "Byte tensor for attn_mask in nn.MultiheadAttention is deprecated."
                    "Use bool tensor instead."
                )
                attn_mask = attn_mask.to(torch.bool)

            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, tgt_len, key_seq_len]:
                    raise ValueError("The size of the 2D attn_mask is not correct.")
            elif attn_mask.dim() == 3:
                if list(attn_mask.size()) != [
                    bsz * self.num_heads,
                    tgt_len,
                    key_seq_len,
                ]:
                    raise ValueError("The size of the 3D attn_mask is not correct.")
            else:
                raise ValueError(
                    "attn_mask's dimension {} is not supported".format(attn_mask.dim())
                )
            # attn_mask's dim is 3 now.

        # convert ByteTensor key_padding_mask to bool
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "Byte tensor for key_padding_mask in nn.MultiheadAttention"
                "is deprecated. Use bool tensor instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        if self.add_bias_kv:
            k = self.seq_bias_k(k)
            v = self.seq_bias_v(v)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat(
                [
                    k,
                    torch.zeros(
                        (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                    ),
                ],
                dim=1,
            )
            v = torch.cat(
                [
                    v,
                    torch.zeros(
                        (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                    ),
                ],
                dim=1,
            )
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ]

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_output_weights += attn_mask

        if key_padding_mask is not None:
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn_output_weights = attn_output_weights.view(
                bsz * self.num_heads, tgt_len, src_len
            )

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, head_dim]

        if self.batch_first:
            attn_output = (
                attn_output.view(bsz, self.num_heads, tgt_len, head_dim)
                .transpose(1, 2)
                .contiguous()
                .view(bsz, tgt_len, embed_dim)
            )
        else:
            attn_output = (
                attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
            )
        attn_output = self.out_proj(attn_output)

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(
                bsz, self.num_heads, tgt_len, src_len
            )
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

    def unsqueeze_0_2(self, t):
        return torch.unsqueeze(torch.unsqueeze(t, 0), 0)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        destination_alter = OrderedDict()
        if len(prefix.split(".")) > 2:
            alter_key = ".".join(prefix.split(".")[:-2]) + ".emb.weight"
        else:
            alter_key = "emb.weight"
        if alter_key in destination:
            destination_alter[alter_key] = destination[alter_key]

        self._save_to_state_dict(destination, prefix, keep_vars)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(
                    destination=destination,
                    prefix=prefix + name + ".",
                    keep_vars=keep_vars,
                )

        if (self.kdim == self.embed_dim) and (self.vdim == self.embed_dim):
            destination_alter[prefix + "in_proj_weight"] = torch.cat(
                (
                    destination[prefix + "qlinear.weight"],
                    destination[prefix + "klinear.weight"],
                    destination[prefix + "vlinear.weight"],
                ),
                0,
            )
        else:
            destination_alter[prefix + "q_proj_weight"] = destination[
                prefix + "qlinear.weight"
            ]
            destination_alter[prefix + "k_proj_weight"] = destination[
                prefix + "klinear.weight"
            ]
            destination_alter[prefix + "v_proj_weight"] = destination[
                prefix + "vlinear.weight"
            ]

        if (
            (prefix + "qlinear.bias") in destination
            and (prefix + "klinear.bias") in destination
            and (prefix + "vlinear.bias") in destination
        ):
            destination_alter[prefix + "in_proj_bias"] = torch.cat(
                (
                    destination[prefix + "qlinear.bias"],
                    destination[prefix + "klinear.bias"],
                    destination[prefix + "vlinear.bias"],
                ),
                0,
            )

        if self.add_bias_kv:
            destination_alter[prefix + "bias_k"] = self.unsqueeze_0_2(
                destination[prefix + "seq_bias_k.bias"]
            )
            destination_alter[prefix + "bias_v"] = self.unsqueeze_0_2(
                destination[prefix + "seq_bias_v.bias"]
            )

        destination_alter[prefix + "out_proj.weight"] = destination[
            prefix + "out_proj.weight"
        ]
        if (prefix + "out_proj.bias") in destination:
            destination_alter[prefix + "out_proj.bias"] = destination[
                prefix + "out_proj.bias"
            ]

        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination_alter, prefix, local_metadata)
            if hook_result is not None:
                destination_alter = hook_result

        return destination_alter
