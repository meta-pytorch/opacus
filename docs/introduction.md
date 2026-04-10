---
id: introduction
title: Introduction
---

Opacus is a library that enables training PyTorch models with differential privacy. It supports training with minimal code changes required on the client, has little impact on training performance, and allows the client to online track the privacy budget expended at any given moment.

Please refer to [this post](https://ai.facebook.com/blog/introducing-opacus-a-high-speed-library-for-training-pytorch-models-with-differential-privacy/) to read more about Opacus.

## Target audience
Opacus is aimed at two target audiences:

1. ML practitioners will find this to be a gentle introduction to training a model with differential privacy as it requires minimal code changes.
2. Differential Privacy scientists will find this easy to experiment and tinker with, allowing them to focus on what matters.

## Model Compatibility

Opacus supports two modes for integrating with your PyTorch models:

**Wrapped mode (default):** Opacus wraps your model in a `GradSampleModule` to compute per-sample gradients. This works well for most models but can cause issues with:
- Type checking (`isinstance()` fails, e.g., HuggingFace Transformers)
- State dict compatibility (`_module.` prefix added to keys)

**Non-wrapping mode:** Set `wrap_model=False` to attach hooks directly to your model without wrapping. This preserves model type, keeps clean state dicts, and provides better compatibility with transformer models. Requires manual cleanup via the returned `hooks.cleanup()` when done.

See the [non-wrapping mode tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/non_wrapping_mode.ipynb) for details.
