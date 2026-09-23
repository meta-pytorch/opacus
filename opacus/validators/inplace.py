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

"""Utilities to make a model free of in-place operations.

Opacus' hooks-based grad_sample modes attach ``register_full_backward_hook`` to
each trainable module. That wraps the module's output in a custom autograd
``Function`` whose result is a view, and autograd forbids in-place writes to such
a view -- hence ``RuntimeError: Output 0 of BackwardHookFunction is a view and is
being modified inplace``. This module removes the two sources of in-place writes:
module-level in-place activations (the ``inplace`` flag) and functional/tensor
in-place ops baked into ``forward`` (e.g. a ResNet's ``out += identity``).
"""

import logging
import operator
import warnings

import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def disable_inplace(module: nn.Module) -> None:
    """Recursively turn off the ``inplace`` flag on every submodule that has it.

    Handles module-based in-place activations (nn.ReLU(inplace=True), SiLU,
    Hardtanh, Dropout, ...). Does NOT handle functional/tensor in-place ops baked
    into a forward() method (e.g. ResNet's ``out += identity``) -- see
    fix_inplace_operations.
    """
    for child in module.modules():
        if getattr(child, "inplace", False):
            child.inplace = False


def _is_inplace_name(name: str) -> bool:
    """A single trailing underscore marks an in-place op (``add_``), but not a dunder."""
    return name.endswith("_") and not name.endswith("__")


def _is_inplace_function(target: object) -> bool:
    """True if a ``call_function`` target is an in-place op (trailing-underscore name).

    Excludes ``operator`` builtins whose trailing underscore only avoids a Python
    keyword and are out-of-place (``operator.and_``, ``operator.or_``, ...): for
    those, ``getattr(operator, name)`` is the target itself.
    """
    name = getattr(target, "__name__", "")
    if not _is_inplace_name(name):
        return False
    return getattr(operator, name, None) is not target


def _outofplace_function(target: object) -> object:
    """Map a functional in-place op (``torch.relu_``) to its out-of-place twin.

    Looks the stripped name up in ``torch`` and ``torch.nn.functional`` so both
    ``torch.relu_`` and ``F.relu_`` resolve. Returns None if no twin exists.
    """
    if not _is_inplace_function(target):
        return None
    stripped = target.__name__[:-1]
    for namespace in (torch, F):
        candidate = getattr(namespace, stripped, None)
        if callable(candidate):
            return candidate
    return None


def _rewrite_zero(node: fx.Node) -> None:
    # x.zero_() -> torch.zeros_like(x)
    node.op = "call_function"
    node.target = torch.zeros_like
    node.args = (node.args[0],)
    node.kwargs = {}


def _rewrite_fill(node: fx.Node) -> None:
    # x.fill_(v) -> torch.full_like(x, v)
    value = node.args[1] if len(node.args) > 1 else node.kwargs["value"]
    node.op = "call_function"
    node.target = torch.full_like
    node.args = (node.args[0], value)
    node.kwargs = {}


def _rewrite_copy(node: fx.Node) -> None:
    # x.copy_(src) -> src.expand_as(x).type_as(x).clone(). copy_ broadcasts src to x's
    # shape, casts it to x's dtype/device, and writes into x's own storage. type_as(x)
    # reproduces that dtype/device cast (the concrete dtype is unknown while rewriting,
    # so it is read from x at runtime); without it a cross-dtype copy_ would leak src's
    # dtype downstream. .clone() gives the result independent, contiguous storage so it
    # behaves like copy_ rather than aliasing src.
    self_node = node.args[0]
    src = node.args[1] if len(node.args) > 1 else node.kwargs["src"]
    node.target = "expand_as"
    node.args = (src, self_node)
    node.kwargs = {}
    graph = node.graph
    with graph.inserting_after(node):
        type_as_node = graph.call_method("type_as", args=(node, self_node))
    with graph.inserting_after(type_as_node):
        clone_node = graph.call_method("clone", args=(type_as_node,))
    node.replace_all_uses_with(clone_node)
    # replace_all_uses_with also redirected the new nodes' own inputs to clone_node;
    # point them back at their real upstream nodes.
    type_as_node.args = (node, self_node)
    clone_node.args = (type_as_node,)


# In-place methods whose out-of-place form is not just the name minus "_".
_METHOD_SPECIAL_CASES = {
    "zero_": _rewrite_zero,
    "fill_": _rewrite_fill,
    "copy_": _rewrite_copy,
}


def _is_inplace_node(node: fx.Node) -> bool:
    """True if ``node`` performs an in-place write (method, function, or out=).

    Augmented-assignment operators (``out += x``) are NOT covered here: ``torch.fx``
    functionalizes them to their out-of-place form while tracing, so they never
    appear as in-place nodes -- the recompiled graph is already out-of-place. Note
    that some out-of-place ``operator`` builtins carry a trailing underscore to
    avoid Python keywords (``operator.and_``, ``operator.or_``); those are excluded
    by ``_is_inplace_function``, not the raw trailing-underscore name check.
    """
    if node.op not in ("call_function", "call_method"):
        return False
    if node.kwargs.get("out") is not None:
        return True
    if node.op == "call_function":
        return _is_inplace_function(node.target)
    return isinstance(node.target, str) and _is_inplace_name(node.target)


def _try_rewrite(node: fx.Node) -> bool:
    """Rewrite an in-place node to its out-of-place form. Return True if rewritten.

    Returns False (leaving the node as-is) when rewriting is unsafe: ``out=`` buffer
    writes, an in-place method/function with no known equivalent, or a node whose
    mutated tensor is also read elsewhere (rewriting out-of-place would hide the
    mutation from those readers and silently change outputs).
    """
    if node.kwargs.get("out") is not None:
        # The result flows through the caller's buffer, not this node's return, so
        # we cannot drop ``out=`` without rewiring downstream users.
        return False

    # An in-place op writes into its first argument. If that tensor is read by any
    # node other than this one, rewriting to the out-of-place form hides the
    # mutation from those readers and silently changes outputs. Leave such nodes in
    # place -- their aliasing semantics are preserved and they get reported.
    self_node = node.args[0] if node.args else None
    if isinstance(self_node, fx.Node) and len(self_node.users) > 1:
        return False

    if node.op == "call_function":
        out_fn = _outofplace_function(node.target)
        if out_fn is not None:
            node.target = out_fn
            return True
        return False

    # call_method
    special = _METHOD_SPECIAL_CASES.get(node.target)
    if special is not None:
        special(node)
        return True
    stripped = node.target[:-1]
    # require a callable twin: e.g. requires_grad_ strips to the property
    # requires_grad, which must not be turned into a method call.
    if callable(getattr(torch.Tensor, stripped, None)):
        node.target = stripped
        return True
    return False


def fix_inplace_operations(module: nn.Module) -> nn.Module:
    """Return a module with functional/tensor in-place operations made out-of-place.

    The module is symbolically traced with ``torch.fx`` and recompiled, which
    accomplishes the rewrite in two ways:

      - augmented-assignment operators are functionalized by tracing itself
        (``out += x`` becomes ``out = out + x`` in the recompiled graph), and
      - in-place methods/functions are rewritten explicitly: ``x.add_()`` ->
        ``x.add()``, ``torch.relu_`` / ``F.relu_`` -> ``torch.relu``, and the
        special cases ``x.zero_()`` -> ``torch.zeros_like(x)``, ``x.fill_(v)`` ->
        ``torch.full_like(x, v)``, ``x.copy_(src)`` -> ``src.expand_as(x).clone()``.

    An in-place method/function is rewritten only when it is safe: its result is
    consumed (so the out-of-place value flows downstream) AND a known out-of-place
    form exists. In-place ops that are left untouched keep their original semantics
    via tensor aliasing and only fail if their tensor is wrapped by an Opacus
    backward hook; they are reported via a warning that distinguishes ops with no
    known out-of-place form from ops whose result is unused.

    If the module cannot be symbolically traced (e.g. data-dependent control flow,
    as in attention/transformer blocks), this is logged at INFO and the module is
    returned unchanged. On a successful trace the returned object is an
    ``fx.GraphModule``.

    Args:
        module: The module to rewrite. Module-level ``inplace`` flags are expected
            to have been cleared already (see :func:`disable_inplace`).

    Returns:
        The rewritten ``fx.GraphModule``, or the original ``module`` if tracing failed.
    """
    try:
        graph_module = fx.symbolic_trace(module)
    except Exception:
        logger.info(
            "If the model has in-place operations, "
            "these may cause errors during the backward pass with Opacus. If so, consider rewriting the model "
            "to replace those in-place operations with out-of-place ones. "
        )
        return module

    aliased = []  # zero users: side effect only observed via tensor aliasing
    out_buffer = (
        []
    )  # writes into a caller-provided out= buffer; can't drop without rewiring
    unrewritable = []  # has users but no known out-of-place twin / self read elsewhere
    for node in graph_module.graph.nodes:
        if not _is_inplace_node(node):
            continue
        # Zero users means symbolic_trace did not thread the mutation into the
        # dataflow; rewriting out-of-place would discard it, so leave it in place
        # (codegen re-emits it and the aliased write is still observed downstream).
        if len(node.users) == 0:
            aliased.append(node.format_node())
        elif node.kwargs.get("out") is not None:
            out_buffer.append(node.format_node())
        elif not _try_rewrite(node):
            unrewritable.append(node.format_node())

    if aliased or out_buffer or unrewritable:
        details = []
        if unrewritable:
            details.append("no safe out-of-place form for: " + "; ".join(unrewritable))
        if out_buffer:
            details.append(
                "result written into a caller-provided out= buffer, dropping out= "
                "would require rewiring downstream users: " + "; ".join(out_buffer)
            )
        if aliased:
            details.append(
                "result unused so tracing dropped the rewrite, the in-place write "
                "is still observed via aliasing: " + "; ".join(aliased)
            )
        warnings.warn(
            "fix_inplace_operations left some in-place ops unchanged. They are fine "
            "unless their tensor is wrapped by an Opacus backward hook, in which "
            "case autograd raises at runtime.\n  " + "\n  ".join(details),
        )

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
