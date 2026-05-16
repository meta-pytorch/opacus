# Changelog

## v1.6.0

### New features

#### Better interoperability with modern training stacks
* Add non-wrapping mode for better compatibility with Transformers, Accelerate, and libraries that expect the original module hierarchy (`wrap_model=False`) (#794)
* Add arithmetic operations support to `DPTensorFastGradientClipping`, making it easier to integrate Opacus with custom loss compositions and external trainers (#805)

#### Distributed and large-model training
* Add support for Fully Sharded Data Parallel (FSDP) training, including a tutorial and a new example (#761,#772,#781,#782)
* Add support for mixed and low precision training (#764)
* Add 1D tensor parallelism support for fast gradient clipping, together with toy and Llama examples; this support is currently beta (#776)

#### Others
* Add ability to register custom noise accountants (#784)

### Bug fixes
* Fix epsilon/noise accounting when using adaptive gradient clipping  (#807, #779)
* Fix fast gradient clipping when using `ignore_index` masking, so ignored tokens do not affect the reduced loss incorrectly (#808)
* Replace empty-batch handling inside `DPDataLoader` with a structure-aware approach, fixing failures for custom batch structures under Poisson sampling (#806)
* Treat `IAccountant.mechanism` correctly during `state_dict` handling (#778)

### Compatibility
* Require `torch>=2.6.0` (#770)

## v1.5.4

### New features

#### Improvments
* Support of `register_full_backward_hook` (#720, #750)
* Add hook function of `RMSNorm` (#755)
* Numpy 2.0 support (#746)

### Bug fixes
* Fix failing tests (#727, #739)
* Fix lint issues (#741, #760)
* Fix the norm calculation of bias under ghost clipping (#751)
* Fix 'to_standard_module' under ghost clipping (#754)
* Fix the return type of 'make_private' (#759)

### Miscellaneous
* Copyright for DISK (#719)
* Improve the logging system (#735)
* Update the tutorials (#733, #745)
* Minor change on the ghost clipping optimizer (#756)
* Warning about restriction for adaptive clipping (#758)

## v1.5.3

### New features

#### Improvments to ghost clipping
* The interface for ghost clipping is now similar to that of PyTorch and vanilla DP-SGD (#668)
* Updated a tutorial on training language models with DP-SGD to include ghost clipping (#667) and LoRA (#698)
* Added adaptive clipping support for ghost clipping (#711)
* Add ghost clipping support for embedding layers (#694)
* Support generative NLP tasks with ghost clipping (#722)
* Add functionality to access per-sample gradients with ghost clipping (#724)

#### Enabling more external contributions
* Added a research folder for external contribution of promising new methods for PPML (#700)
* DP-SGD optimizers with Kalman filters are now available in the research folder (#706)
* Made it easier to define custom extensions of PrivacyEngine (#703, #704, #710)

### Bug fixes
* Fix the clipping operation for ghost clipping when using the PrivacyEngine interface (#664)
* Fix issue with ghost clipping and BatchMemoryManager
* Add `strict` and `force_functorch` parameters in initaliziation of `GradSampleModuleFastGradientClipping` (#675)
* Fix failing tests (e.g., #726, #713, #727, #674)

### Miscellaneous
* Switch from  testing with CircleCI to GithubActions CI (#701)
* Website and Github improvements (#723, #721, #677, #712)
* Added multi-gpu test for ghost clipping (#665)

## v1.5.2

### New features
* Add a function of "double_backward" simplifying the training loop (#661)

### Bug fixes
* Fix issue with setting of param_group for the DPOptimizer wrapper (issue 649) (#660)
* Fix issue of DDP optimizer for FGC. The step function incorrectly called "original_optimizer.original_optimizer" (#662)
* Replace "opt_einsum.contract" by "torch.einsum"(#663)

## v1.5.1

### Bug fixes
* Make the import of opt_einsum.contract (linear.py) explicit (#658)

## v1.5

### New features
Fast Gradient Clipping and Ghost Clipping (#656)

### Bug fixes
* Fix gradient shape error for DPMultiheadAttention (issue 650) (#651)
* Pass kwargs from make_private to _prepare_optimizer (#648)
* Fix BatchMemoryManager length (#641)
* Fix GPU-CPU device mismatch error in util filter_dilated_rows (#633)
* Fix Opacus's runtime error with an empty batch (issue 612) (#631)

## v1.4.1

### Bug fixes
* Fix DP MultiheadAttention (#598)
* Fix: make prv accountant robust to larger epsilons (#606)
* Fix the corner case when the optimizer has no trainable parameters (#619)

## v1.4

Highlight: Upgraded to PyTorch 1.13+ as required dependency

### New features
* Added clipping schedulers (#556)
* Util to check per sample gradients (#532)

### Bug fixes
* Align DataLoader interface with vanilla PyTorch (#543)
* Fix GDP accountant epsilon retrieval changing internal state (#541)
* Add option to specify number of steps in UniformSampler (#550)
* Fix privacy computation script (#565)


## v1.3

### New features
* Implement the `PRVAccountant` based on the paper [Numerical Composition of Differential Privacy](https://arxiv.org/abs/2106.02848) (#493)
* Support `nn.EmbeddingBag` (#519)

### Bug fixes
* Fix benchmarks (#503, #507, #508)
* Align `make_private_with_epsilon` with `make_private` (#509, #526)
* Test fixes (#513, #515, #527, #533)
* Summed discriminator losses to perform one backprop step (#474)
* Fixed issue with missing argument in MNIST example (#520)
* Functorch gradients: investigation and fix (#510)
* Support empty batches (#530)

## v1.2

### New ways to compute per sample gradients
We're glad to present Opacus v1.2, which contains some major updates to per sample gradient computation mechanisms
and includes all the good stuff from the recent PyTorch releases.
* Functorch - per sample gradients for all
* ExpandedWeights - yet another way to compute per sample gradients
* See [Release notes](https://github.com/pytorch/opacus/releases/tag/v1.2.0)
  and [GradSampleModule README](https://github.com/pytorch/opacus/blob/main/opacus/grad_sample/README.md)
  for detailed feature explanation

### Other improvements
* Fix `utils.unfold2d` with non-symmetric pad/dilation/kernel_size/stride (#443)
* Add support for "same" and "valid" padding for hooks-based grad sampler for convolution layers
* Improve model validation to support frozen layers and catch copied parameters (#489)
* Remove annoying logging from `set_to_none` (#471)
* Improved documentation (#480, #478, #482, #485, #486, #487, #488)
* Imtegration test improvements (#407, #479, #481. #473)


## v1.1.3
### Bug fixes
* Support layers with a mix of frozen and learnable parameters (#437)
* Throw an error when params in optimizer are not the same as that of module's in make_private (#439)
* Fix unfold2d and add test (#443)

### Miscellaneous
* Fix typos in DDP tutorial (#438)
* Replace torch einsum with opt_einsum (#440)

## v1.1.2
### Bug fixes
* Support tied parameters (#417)
* Fix callsite sensitiveness of `zero_grad()` (#422, #423)
* Improve microbenchmark argument parsing and tests (#425)
* Fix opacus nn.functional import (#426)
### Miscellaneous
* Add microbenchmarks (#412, #416)
* Add more badges to readme (#424)

## v1.1.1
### Bug fixes
* Fix accountant when using number of steps instead of epochs
* Add params check when converting BatchNorm to GroupNorm (#390)
* Fix typo in gdp accountant mechanism name (#386)
* Fix linter errors (#392)
* Add friendly and detailed message for unsupported layers (#401)
* Run linter on nightly workflow (#399)
* Add warning for Gaussian DP accounting (#400)
* Clone replacement modules on the same device as original (#356)
* Implementing 3D dilation (#408)
* fix(batch_memory_manager): Ensures split_idxs use native python types (#410)
### Miscellaneous
* Migrate nightly CircleCI flows to scheduled pipelines (#402)
* Migrate from ubuntu 16.04 to 20.04 on CircleCI (#403)

## v1.1.0
### New Feature
* Add support for GDP accounting in get_noise_multiplier (#303)
### Bug fixes
* Conservative search for target epsilon in get_noise_multiplier (#348)
* Warn and ignore "drop_last" when set in DPDataLoader (#357)
* Fix per-layer clipping in distributed (#347)
### Miscellaneous
* Update code of conduct and file headers
* Add "Support Ukraine" banner to opacus website homepage
* Lint fixes

## v1.0.2
### Bug fixes
* DPOptimizer
  * Passes through `.defaults` field to match pytorch Optimizer (#329)
  * Better exception message in `.step()` when p.grad_sample=None (#331)
  * Correct `closure` call after applying DP noise (#330)
* Proper gradient scaling in DDP mode
* Corrections of typos and errors in tutorials
### Miscellaneous
* Opacus can be installed with conda: added recipe in conda-forge (#326)
* Formatting change in accordance with black-22.1.0

## v1.0.1
### Bug fixes
* Hidden states of RNN is passed to device (#314)
* Validate and fix trainable modules only (#316)
### Miscellaneous
* Minor corrections and typo fixes in links, documentation, and tutorials.

## v1.0.0
* This release packs in lot of new features and bug fixes, and most importantly, also brings forth new APIs that are simpler, more modular, and easily extensible.
* We have bumped up the major version number from 0 to 1 and have introduced breaking changes. However, the major version bump also indicates a step-function upgrade in the capabilities.
* See [Release notes](https://github.com/pytorch/opacus/releases/tag/v1.0.0] and [Migration Guide](https://github.com/pytorch/opacus/blob/main/Migration_Guide.md) for more details about the changes.
* PR #273 contains the pointers to all the commits and PRs that went into this release.

## v0.15.0
### New Features
* DDP support for faster distributed training (#196)
* Support of GRU and RNN; refactored LSTM implementation (#222)
* PyTorch Lightning Demo (#244)
### Bug fixes
* Improve nn.Linear grad sampler memory consumption (#192)
* Update Opacus to stop using deprecated torch.set_deterministic (#197)
* Fix optimizer.step after engine.detach()
* Test fixes
### Miscellaneous
* Better validation error reporting (#199)
* grad sampler type checking (#241)

## v0.14.0
### New features
* Major refactoring - per-sample gradient computation is separated into its own module - GradSampleModule (#175)
* Improved RDP to (eps, delta)-DP conversion (#162)
* Multi-GPU support (#166)
### Bug fixes
* Handle empty batches in Poisson sampling (#164)
* Fixed memory leak from no_grad execution (#180)

## v0.13.0
### New features
* PackedSequence support for DPLSTM (#150) (thanks @touqir14 !)
### Miscellaneous
* Pytest moved to dev installation (#144)

## v0.12.0
This version introduces a **mildly-breaking change**: the privacy engine will now support sampling with variable batch size, just like in the Abadi et al. paper. To accommodate this feature, we have made `batch_size` a kwarg (no longer positional). We are also enforcing that all kwargs must not be specified positionally. If you had code that passed kwargs positionally, you will find an error (which will be very simple to fix).
### New features
* Enforce kwargs to Privacy Engine (#136).
* Fix batch construction and privacy engine (#128). (thanks @ConstanceBeguier!)
* Compute required sigma to reach (epsilon, delta) budget (#126)
* Friendly user message for unused parameters (#118).
* Print helpful message when models are not in train mode (#113)
### Bug fixes
* Now the Opacus package has a `__version__` attribute.
* Fix immer security issue, fix website errors
* Updated setup.py version requirements to support 3.6.8 for Windows (#108) (thanks @madhavajay!)
### Miscellaneous
* Rewrote the grad_sample tests to use Hypothesis (#125). (thanks @touqir14!)

## v0.11.0
### New features
* Extend DPLSTM to support multilayer, dropout (#101)
* Modifications to Char LSTM name classification example
* Introduce issue templates for GitHub (#102)
* Added support for Conv3D layers
### Bug fixes
* Linter fixes for Conv3D (#105)
### Miscellaneous
* Make TorchCSPRNG an optional dependency (#106)
* Removed unnecessary calls to zero_grad from examples and tutorials (#96)

## v0.10.1
### Bug fixes
* Fix PyPI deployment (#91).
### Miscellaneous
* Refactor grad sample tests (#90).
* Avoid storing activations in certain scenarios (#87)

## v0.10.0
### New features
* Reimplemented the Embedding layer, making it 9x faster with lower memory footprint (#73).
* Reimplemented the DPLSTM layer, making it 2x faster with lower memory footprint.
* Extended our Conv support to grouped convolutions (#78).
### Bug fixes
* Small fixes to clipping logic (#45).
### Miscellaneous
* Changed docstring style from numpy -> Google.
* Throw an error if sample rate > 1 in privacy engine.
* Migrated our IMDB example from TorchText -> HuggingFace (#85).
* Added PRNG shuffling to our examples.

## v0.9.1
### Bug fixes
* Compatibility with Python 3.6 (Minimum required version changed from 3.7 to 3.6.9).
* Allow DP-LSTM to have null init.

## v0.9.0
### New Features
* Initial commit.
