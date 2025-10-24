# Grad Samples

Computing per sample gradients is an integral part of Opacus framework. We strive to provide out-of-the-box support for
wide range of models, while keeping computations efficient.

We currently provide three independent approaches for computing per sample gradients:

1. **Hooks-based `GradSampleModule`** (stable, wraps the model)
2. **`GradSampleController`** (stable, no model wrapping - recommended for transformers)
3. **`GradSampleModuleExpandedWeights`** (beta, based on PyTorch 1.12+ functionality)

Each implementation comes with its own set of limitations and benefits.

**TL;DR:**
- Use `GradSampleModule` (`grad_sample_mode="hooks"`) for stable implementation with standard models
- Use `GradSampleController` via `PrivacyEngineGradSampleController` for transformer models and when you need direct model access without wrapping
- Use `GradSampleModuleExpandedWeights` (`grad_sample_mode="ew"`) if you want to experiment with better performance
- Use `grad_sample_mode="functorch"` if your model has unsupported layers

Please report any strange errors or unexpected behaviour to us!

## GradSampleController approach (No Model Wrapping)
- Controller class: ``opacus.grad_sample.GradSampleController``
- Privacy Engine: ``opacus.privacy_engine_gsc.PrivacyEngineGradSampleController``
- Usage: Use `PrivacyEngineGradSampleController` instead of `PrivacyEngine`

**Recommended for transformer models and when model wrapping causes issues.**

Computes per-sample gradients by attaching hooks directly to model parameters without wrapping the model in a
`GradSampleModule`. This approach:

- ✅ Preserves model type (e.g., `isinstance(model, BertModel)` remains `True`)
- ✅ No `_module.` prefix in state_dict
- ✅ Direct access to model attributes (no attribute forwarding needed)
- ✅ Better compatibility with HuggingFace transformers and models with custom `__getattr__`
- ✅ Same grad sampler methods as `GradSampleModule`

See [CONTROLLER_BASED_PRIVACY_ENGINE.md](../../docs/CONTROLLER_BASED_PRIVACY_ENGINE.md) for detailed documentation.

## Hooks-based approach (Model Wrapping)
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="hooks"`

Computes per-sample gradients for a model using backward hooks. It requires custom grad sampler methods for every
trainable layer in the model. We provide such methods for most popular PyTorch layers. Additionally, client can
provide their own grad sampler for any new unsupported layer (see [tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb))

## Functorch approach
- Model wrapping class: ``opacus.grad_sample.grad_sample_module.GradSampleModule (force_functorch=True)``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="functorch"`

[functorch](https://pytorch.org/functorch/stable/) is JAX-like composable function transforms for PyTorch.
With functorch we can compute per-sample-gradients efficiently by using function transforms. With the efficient
parallelization provided by `vmap`, we can obtain per-sample gradients for any function function (i.e. any model) by 
doing essentially `vmap(grad(f(x)))`. 

Our experiments show, that `vmap` computations in most cases are as fast as manually written grad samplers used in 
hooks-based approach.

With the current implementation `GradSampleModule` will use manual grad samplers for known modules (i.e. maintain the
old behaviour for all previously supported models) and will only use functorch for unknown modules.

With `force_functorch=True` passed to the constructor `GradSampleModule` will rely exclusively on functorch. 

## ExpandedWeigths approach
- Model wrapping class: ``opacus.grad_sample.gsm_exp_weights.GradSampleModuleExpandedWeights``
- Keyword argument for ``PrivacyEngine.make_private()``: `grad_sample_mode="ew"`

Computes per-sample gradients for a model using core functionality available in PyTorch 1.12+. Unlike hooks-based
grad sampler, which works on a module level, ExpandedWeights work on the function level, i.e. if your layer is not
explicitly supported, but only uses known operations, ExpandedWeights will support it out of the box.

At the time of writing, the coverage for custom grad samplers between ``GradSampleModule`` and ``GradSampleModuleExpandedWeights``
is roughly the same.

## Comparative analysis

Please note that these are known limitations and we plan to improve Expanded Weights and bridge the gap in feature completeness


| xxx                          | GradSampleModule (Hooks) | GradSampleController | Expanded Weights | Functorch    |
|:----------------------------:|:------------------------:|:-------------------:|:----------------:|:------------:|
| Required PyTorch version     | 1.8+                     | 1.8+                | 1.13+            | 1.12 (to be updated) |
| Development status           | Deprecated mechanism     | ✅ Stable           | Beta             | Beta         |
| Model wrapping               | ✅ Wraps model           | ✅ No wrapping      | ✅ Wraps model   | ✅ Wraps model |
| Runtime Performance†          | baseline                | baseline            | ✅ ~25% faster   | 🟨 0-50% slower |
| Transformer compatibility    | 🟨 May have issues      | ✅ Excellent        | 🟨 May have issues | 🟨 May have issues |
| State dict compatibility     | 🟨 `_module.` prefix    | ✅ Clean keys       | 🟨 `_module.` prefix | 🟨 `_module.` prefix |
| Type preservation            | ❌ Model wrapped        | ✅ Model unchanged  | ❌ Model wrapped | ❌ Model wrapped |
| Any DP-allowed†† layers       | Not supported          | Not supported       | Not supported    | ✅ Supported |
| Most popular nn.* layers     | ✅ Supported            | ✅ Supported        | ✅ Supported     | ✅ Supported  |
| torchscripted models         | Not supported           | Not supported       | ✅ Supported     | Not supported |
| Client-provided grad sampler | ✅ Supported            | ✅ Supported        | Not supported    | ✅ Not needed |
| `batch_first=False`          | ✅ Supported            | ✅ Supported        | Not supported    | ✅ Supported  |
| Recurrent networks           | ✅ Supported            | ✅ Supported        | Not supported    | ✅ Supported  |
| Padding `same` in Conv       | ✅ Supported            | ✅ Supported        | Not supported    | ✅ Supported  |
| Empty poisson batches        | ✅ Supported            | ✅ Supported        | Not supported    | Not supported  |

† Note, that performance differences are unstable and can vary a lot depending on the exact model and batch size.
Numbers above are averaged over benchmarks with small models consisting of convolutional and linear layers.
Note, that performance differences are only observed on GPU training, CPU performance seem to be almost identical
for all approaches.

†† Layers that produce joint computations on batch samples (e.g. BatchNorm) are not allowed under any approach    

