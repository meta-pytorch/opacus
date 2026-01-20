# Grad Samples

Computing per sample gradients is an integral part of Opacus framework. We strive to provide out-of-the-box support for
wide range of models, while keeping computations efficient.

We currently provide two independent approaches for computing per sample gradients: hooks-based ``GradSampleModule``
(stable implementation, exists since the very first version of Opacus) and ``GradSampleModuleExpandedWeights``
(based on a beta functionality available in PyTorch 1.12).

Each of the two implementations comes with it's own set of limitations, and we leave the choice up to the client
which one to use.

``GradSampleModuleExpandedWeights`` is currently in early beta and can produce unexpected errors, but potentially
improves upon ``GradSampleModule`` on performance and functionality.

**TL;DR:** If you want stable implementation, use ``GradSampleModule`` (`grad_sample_mode="hooks"`).
If you want to experiment with the new functionality, you have two options. Try 
``GradSampleModuleExpandedWeights``(`grad_sample_mode="ew"`) for better performance and `grad_sample_mode=functorch` 
if your model is not supported by ``GradSampleModule``. 

Please switch back to ``GradSampleModule``(`grad_sample_mode="hooks"`) if you encounter strange errors or unexpexted behaviour.
We'd also appreciate it if you report these to us

## Hooks-based approach
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

## Non-wrapping mode (Hooks without wrapper)

Introduced in version 1.4.0, Opacus supports hooks-based grad sample computation without wrapping the model.

### Why use non-wrapping mode?

By default, Opacus wraps the model in a `GradSampleModule` wrapper, which can cause compatibility issues with some architectures:
- Type checking: `isinstance(model, MyModel)` returns `False` after wrapping.
- State dict keys: Wrapped models add a `_module.` prefix to all parameter names.
- Attribute access: Some models with custom `__getattr__` (e.g., HuggingFace Transformers) may not work as expected.
- Introspection: Tools that inspect the model structure will see the wrapper instead of the original model.

Non-wrapping mode addresses these issues by attaching hooks directly to model parameters without changing the model object itself.

### Usage

Set `wrap_model=False` in `PrivacyEngine.make_private()`:

```python
from opacus import PrivacyEngine

# Your model - untouched by Opacus
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

privacy_engine = PrivacyEngine()
hooks, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
    wrap_model=False,  # Enable non-wrapping mode
)
# hooks is a GradSampleHooks object for cleanup
# model is the original model instance
```

### Using the model in non-wrapping mode

In non-wrapping mode, the original model remains unchanged and can be used directly for training and evaluation.

```python
# Use the model instance directly
output = model(input)                     # Forward pass
state_dict = model.state_dict()          # Get state dict
model.train()                            # Switch to train mode
torch.save(model.state_dict(), 'model.pt')  # Save checkpoint
```

The `hooks` object returned by `make_private` is used for cleanup. It does not support `nn.Module` methods like `.state_dict()` or `forward()`; the original model should be used for these operations.

### Cleanup

When training is complete, clean up using the hooks object:

```python
# Clean up hooks when done
hooks.cleanup()
```

This removes all hooks and attributes added by Opacus from the model parameters.

### Limitations

- ExpandedWeights support: The `grad_sample_mode="ew"` mode requires overriding `.forward()` and is only available with model wrapping.
- Manual cleanup: Unlike wrapped mode, hooks must be explicitly cleaned up when switching datasets or ending training.

### When to use non-wrapping mode

Use `wrap_model=False` when:
- Working with HuggingFace Transformers or other models with complex `__getattr__` logic.
- Model type checks (`isinstance()`) are required by the pipeline or optimizations.
- Clean state dicts without `_module.` prefixes are preferred.
- The pipeline relies on model type introspection.

Use the default wrapped mode (`wrap_model=True`) when:
- Working with standard models without complex introspection needs.
- Automatic cleanup is preferred (the wrapper is discarded when the model goes out of scope).

See the [non-wrapping mode tutorial](../tutorials/non_wrapping_mode.ipynb) for a complete example.

## Comparative analysis

Please note that these are known limitations and we plan to improve Expanded Weights and bridge the gap in feature completeness


|           Feature            |         Hooks (Wrapped)         | Expanded Weights |           Functorch            |
|:----------------------------:|:-------------------------------:|:----------------:|:------------------------------:| 
|   Required PyTorch version   |              1.8+               |      1.13+       |      1.12 (to be updated)      |
|      Development status      | Underlying mechanism deprecated |       Beta       |              Beta              | 
|      Non-wrapping mode       | Supported (`wrap_model=False`)  |  Not supported   | Supported (`wrap_model=False`) |
|     Runtime Performance†     |            baseline             |   ~25% faster    |          0-50% slower          |
|   Any DP-allowed†† layers    |          Not supported          |  Not supported   |           Supported            |
|   Most popular nn.* layers   |            Supported            |    Supported     |           Supported            | 
|     torchscripted models     |          Not supported          |    Supported     |         Not supported          |
| Client-provided grad sampler |            Supported            |  Not supported   |           Not needed           |
|     `batch_first=False`      |            Supported            |  Not supported   |           Supported            |
|      Recurrent networks      |            Supported            |  Not supported   |           Supported            |
|    Padding `same` in Conv    |            Supported            |  Not supported   |           Supported            |
|    Empty poisson batches     |            Supported            |  Not supported   |         Not supported          |

† Note, that performance differences are unstable and can vary a lot depending on the exact model and batch size. 
Numbers above are averaged over benchmarks with small models consisting of convolutional and linear layers. 
Note, that performance differences are only observed on GPU training, CPU performance seem to be almost identical 
for all approaches.

†† Layers that produce joint computations on batch samples (e.g. BatchNorm) are not allowed under any approach

