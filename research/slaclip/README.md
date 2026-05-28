# SlaClip: Gradient Norm Slacks can be an Indicator for Adaptive Clipping in DP-SGD

SlaClip is a privacy-preserving adaptive clipping method for differentially
private stochastic gradient descent (DP-SGD). It uses the norm budget left
unused by clipping—the *slack*—to release a noisy, binned estimate of the
gradient-norm cumulative distribution function (CDF). That Slack Indicator can
then drive the clipping-threshold update from the SlaClip paper or another
post-processing controller.

This directory is a self-contained research prototype. It does not modify
`PrivacyEngine` or any other Opacus core API.

## Method

For a per-sample gradient `g_i`, clipping threshold `C`, and `K` CDF slots,
SlaClip defines the extended gradient from equations (6)-(8) as

```text
g_i^+ = [Clip_C(g_i); s_i]
lambda = C / sqrt(K)
sqrt(K) * max(C - ||g_i||, 0) = a * lambda + b
s_i = [lambda * 1(a); b; 0], where 0 <= b < lambda.
```

This construction guarantees `||g_i^+||_2 <= C`. Under add/remove adjacency
and the fixed normalization constant `B`, the extended average query therefore
has the same `C / B` L2 sensitivity as the vanilla clipped-gradient query. With
the same sampling rule and noise multiplier, it can use the same per-step
privacy-accounting parameters as DP-SGD.

The first `d` coordinates are exactly the native DP-SGD gradient release. The
implementation computes the gradient and `K` slack coordinates separately to
avoid materializing a `d + K` tensor. Adding independent `N(0, (sigma*C)^2)`
noise to both parts is distributionally identical to one isotropic Gaussian
draw in `d + K` dimensions.

After aggregation and noise, the last `K` coordinates are divided by
`B * lambda`. The resulting `optimizer.slack_indicator` is the paper's noisy,
bin-averaged CDF estimate from equation (11). Its first coordinate describes
norms near `C`; its last coordinate describes norms near zero.

> **Privacy boundary:** only the noisy `slack_indicator` property is a public
> output. Per-sample slack and the unnoised slack aggregate are internal to the
> joint DP-SGD mechanism and must not be released. Computing a separate CDF
> query outside this mechanism would require its own privacy analysis and
> accounting.

### Selecting the slack dimension K

When `num_slots=None` (the default), the optimizer selects `K` using the
99%-confidence SNR rule in equation (36):

```text
K_max = (B / (2 * 2.576 * sigma))^(2/3)
K = max(1, floor(K_max)).
```

Here, `B` is the fixed normalization constant used by Opacus
(`expected_batch_size`), not a realized Poisson batch size or a physical
microbatch size. Table 3 gives the following illustrative practical choices
for representative batch sizes when `sigma=1`:

| B | 128 | 256 | 512 | 1024 | 2048 |
|---:|---:|---:|---:|---:|---:|
| Illustrative practical K | 8 | 10 | 20 | 30 | 50 |

These rounded values show the approximate scale of `K`; they are not treated
as special cases by the implementation. The automatic rule always evaluates
equation (36) directly, giving `K = {8, 13, 21, 34, 54}` for the batch sizes
above when `sigma=1`. Pass a positive integer as `num_slots` to reproduce a
particular practical choice or run an ablation. The selected value is
available as `optimizer.K`.

## Usage

The prototype supports two composable steps:

1. `SlaClipDPOptimizer` jointly releases the DP gradient and private Slack
   Indicator. With no controller, the clipping threshold remains unchanged.
2. `SlaClipController` consumes the released indicator and applies equations
   (28)-(30) to update the threshold. Since this is post-processing of a DP
   release, it adds no privacy cost.

### Prepare the optimizer

Use `PrivacyEngine.make_private()` normally, then replace its `DPOptimizer`
with the research optimizer. The helper below preserves all privacy parameters
and transfers the accountant hook:

```python
from opacus import PrivacyEngine

from research.slaclip.slaclipoptimizer import (
    SlaClipController,
    SlaClipDPOptimizer,
)


privacy_engine = PrivacyEngine()
model, private_optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)


def make_slaclip_optimizer(private_optimizer, *, num_slots=None, controller=None):
    optimizer = SlaClipDPOptimizer(
        private_optimizer.original_optimizer,
        noise_multiplier=private_optimizer.noise_multiplier,
        max_grad_norm=private_optimizer.max_grad_norm,
        expected_batch_size=private_optimizer.expected_batch_size,
        loss_reduction=private_optimizer.loss_reduction,
        generator=private_optimizer.generator,
        secure_mode=private_optimizer.secure_mode,
        num_slots=num_slots,
        clipping_controller=controller,
    )
    optimizer.attach_step_hook(private_optimizer.step_hook)
    return optimizer
```

The noise multiplier passed to `SlaClipDPOptimizer` must remain the same as the
one registered with the accountant.

### Step 1 only: obtain private CDF information

Omit the controller to keep `C` fixed while obtaining the noisy Slack
Indicator:

```python
optimizer = make_slaclip_optimizer(
    private_optimizer,
)

for images, targets in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(images), targets)
    loss.backward()
    optimizer.step()

    private_cdf = optimizer.slack_indicator
    # Use private_cdf only in DP-safe post-processing.
```

Because Gaussian noise is unbounded, individual coordinates can fall outside
`[0, 1]` or fail to be monotone. A downstream method may project or smooth the
released vector as post-processing without additional privacy cost.

### Steps 1 and 2: paper SlaClip

Pass the paper controller to adapt `C` after every release:

```python
optimizer = make_slaclip_optimizer(
    private_optimizer,
    controller=SlaClipController(
        eta=0.5,
        min_clipbound=0.1,
        max_clipbound=50.0,
    ),
)

for images, targets in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(images), targets)
    loss.backward()
    optimizer.step()

    print(optimizer.current_clip)
```

The controller implements

```text
r_t     = clip_[0,1](slack_indicator[K] / C_t)
gamma_t = clip_[0,1](1 - (1 - r_t) / 2)
C_(t+1) = clip_[C_min,C_max](
              C_t * exp(eta * (gamma_t - slack_indicator[1]))
          ).
```

The released indicator contains unbounded Gaussian noise. The controller
therefore projects `r_t` and `gamma_t` onto `[0, 1]`, as specified in the
paper, and bounds the next positive clipping threshold to
`[min_clipbound, max_clipbound]`. It intentionally does not force
`gamma_t - slack_indicator[1]` to be nonnegative: a negative value is the
feedback that decreases an overly large clipping threshold.

A custom callable with signature `(current_clip, slack_indicator) -> next_clip`
can replace `SlaClipController`. Such a method reuses the SlaClip Slack
Indicator but is not the paper's threshold controller and should be described
accordingly.

### Parameters

- `num_slots`: number `K` of CDF bins. `None` automatically selects it from
  equation (36). A positive integer overrides the automatic value.
  Larger values increase resolution but also increase normalized indicator
  noise.
- `clipping_controller`: optional post-processing callable. `None` enables the
  indicator-only mode.
- `eta`: positive multiplicative update step size used by
  `SlaClipController`.
- `min_clipbound`, `max_clipbound`: positive lower and upper bounds for the
  clipping threshold. Their defaults, `0.1` and `50.0`, match Opacus
  `AdaClipDPOptimizer` and the SlaClip experiment configuration.
- All other optimizer arguments have the same meaning as in Opacus
  `DPOptimizer`.

## Limitations

- The prototype supports the standard, non-distributed `DPOptimizer` path. It
  is not registered as a `PrivacyEngine` clipping mode.
- Use it from the repository root through `research.slaclip`; research modules
  are not part of the installed Opacus public API.
- The privacy argument assumes the same sampling rule, normalization constant,
  clipping threshold, noise multiplier, and accountant parameters for the
  gradient and slack parts of the joint release.
- As research code, it is not covered by Opacus public API compatibility
  guarantees.

## Tests

From the repository root:

```bash
python -m pytest research/slaclip -q
```

The tests cover automatic `K` selection, equations (7)-(8), the
extended-gradient norm bound, exact agreement of the first `d` coordinates with
native `DPOptimizer`, indicator-only operation, the paper controller, and empty
Poisson batches.

## Citation

```bibtex
@inproceedings{zou2026slaclip,
  title={{SlaClip}: Gradient Norm Slacks Can Be an Indicator for Adaptive
         Clipping in {DP-SGD}},
  author={Zou, Shuyan and Wang, Shaowei and Zhu, Zhanxing and Li, Jin and
          Dong, Changyu and Sassone, Vladimiro and Wu, Han},
  booktitle={Proceedings of the 43rd International Conference on Machine
             Learning},
  year={2026}
}
```

The authors' reference implementation is available at
[ZsyRock/SlaClip](https://github.com/ZsyRock/SlaClip).
