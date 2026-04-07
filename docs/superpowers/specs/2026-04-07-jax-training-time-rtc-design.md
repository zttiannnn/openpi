# JAX Training-Time RTC Design

## Goal

Add a JAX-only training-time RTC path for `pi05_agileX_thor_jax` that improves chunk handoff smoothness without inference-time autograd guidance, while keeping the current JAX inference-time RTC modes available as baselines.

## Why

`openpi` already has JAX inference-time RTC baselines:

- `raw_paper`
- `executed_overlap_paper`

Those modes improve handoff quality by adding guidance during sampling, but they also add inference-time work and make deployment more complex. The follow-up RTC paper proposes moving most of that work into training by teaching the policy to complete a chunk when an executed prefix is already fixed.

For `openpi`, the first version should stay as close as possible to that claim:

- no model architecture changes
- no PyTorch scope
- minimal runtime surface change
- preserve existing JAX RTC modes for A/B comparison

## Scope

This design covers:

- JAX only
- `pi05_agileX_thor_jax` only
- a new training-time RTC training path
- a new JAX sampling mode that uses prefix clamping instead of autograd guidance
- config and CLI wiring needed to train and evaluate the new checkpoint
- focused tests for training semantics, sampling semantics, and mode wiring

This design does not cover:

- PyTorch RTC
- changes to the underlying `Pi0` / `Pi0.5` architecture
- per-token timestep conditioning
- mixed 20Hz/30Hz deployment tuning in the first checkpoint
- removing or altering existing `raw_paper` / `executed_overlap_paper` behavior

## Chosen Approach

### Recommendation

Implement training-time RTC as prefix freezing plus postfix-only loss.

For each training example, sample a delay `d`, treat the first `d` actions of the ground-truth chunk as an already-executed prefix, keep that prefix clean, noise only the remaining postfix actions, and compute flow-matching loss only on the postfix.

At inference time, add a new JAX RTC mode that does not compute a guidance gradient. Instead, it uses the same runtime context shape already available in the async JAX runner and clamps the first `d` denoised actions to the prior chunk prefix during each denoising step.

### Why this approach

This is the best fit for the current `openpi` JAX stack because:

- it preserves the current `Pi0.5` network structure
- it only changes training and sampling logic
- it reuses the existing async RTC runtime interface (`prev_actions`, `inference_delay`, `execution_horizon`)
- it removes the inference-time backward pass that makes RTC expensive
- it keeps the current JAX RTC baselines intact for comparison

### Rejected alternatives

#### Explicit prefix-mask inputs

Adding `prefix_mask`, `delay_steps`, or dedicated prefix tokens to the model would likely make the conditioning more explicit, but it would break the "no architecture change" goal and increase checkpoint and compatibility risk.

#### Loss-only regularization without prefix freezing

A seam loss or overlap-weighted loss would be easier to bolt on, but it is not the core training-time RTC method and would be a weaker replacement for inference-time RTC.

## Delay Model

Training-time RTC should not sample `d` uniformly. It should use the measured delay distribution from the real async JAX pipeline.

Measured histograms:

- `scripts/debug/Async/delay_histogram_async_fps20.json`
  - warmup sample `d=0`: 1
  - steady-state samples `d=3`: 187 / 187
- `scripts/debug/Async/delay_histogram_async_fps30.json`
  - warmup sample `d=0`: 1
  - steady-state samples `d=4`: 188 / 199
  - steady-state samples `d=5`: 11 / 199

The absolute delay is roughly the same in both runs, about 150ms, but the step count depends on control frequency. Because the target deployment for this first checkpoint is 30Hz, the training sampler should follow the 30Hz histogram and ignore the single warmup `d=0` sample.

First-version training sampler:

- `P(d=4) = 188 / 199 ≈ 0.945`
- `P(d=5) = 11 / 199 ≈ 0.055`

The 20Hz histogram remains useful as a robustness reference, but it should not define the first checkpoint.

## Training Semantics

### Base model behavior today

`Pi0.compute_loss()` currently does standard flow matching over the full action chunk:

- sample `noise`
- sample scalar `time`
- build `x_t = time * noise + (1 - time) * actions`
- predict `v_t`
- regress to `u_t = noise - actions`

The first version of training-time RTC should keep that structure and only change how `x_t` and the loss mask are constructed.

### New training-time RTC behavior

For each sample in the batch:

1. Sample delay `d` from the empirical 30Hz distribution.
2. Split the ground-truth chunk `actions` into:
   - prefix: `actions[:, :d]`
   - postfix: `actions[:, d:]`
3. Sample `noise` and scalar `time` exactly as today.
4. Build a mixed target:
   - `x_t[:, :d] = actions[:, :d]`
   - `x_t[:, d:] = time * noise[:, d:] + (1 - time) * actions[:, d:]`
5. Keep the standard flow target:
   - `u_t = noise - actions`
6. Apply loss only on postfix positions `d:`.

This means the model learns to denoise and complete the future part of the chunk while seeing a clean executed prefix that it must respect.

### Batch shape strategy

Do not introduce variable-length action tensors. Keep fixed `(B, H, D)` shapes and use masks internally:

- `prefix_mask[b, t] = 1` when `t < d_b`
- `postfix_mask[b, t] = 1` when `t >= d_b`

These masks are internal to loss construction. They are not model inputs in the first version.

### Important model limitation

`pi0.5` currently uses one global flow timestep per chunk, not per action token. That means the prefix is clean while the postfix is noised under a shared scalar `time`. This is a mild mismatch relative to a more explicit per-token formulation, but it is acceptable for v1 because the same rule is used consistently in both training and sampling and avoids any architecture change.

## Inference Semantics

### New mode

Add a new JAX RTC mode:

- `training_time_prefix`

This mode is only intended for checkpoints trained with training-time RTC.

### Runtime inputs

Reuse the current JAX async RTC context:

- `prev_actions`
- `inference_delay`
- `execution_horizon`

No executed-space guidance inputs are required for this mode, because there is no autograd guidance objective. `execution_horizon` may remain in the runtime context for interface compatibility, but the `training_time_prefix` mode does not use it to build a guidance loss.

### Sampling behavior

During sampling, each denoising step should:

1. Run the normal denoise step to get `v_t`.
2. Update `x_t` as usual.
3. Clamp the first `d = clip(inference_delay, 0, action_horizon)` action steps back to the prior chunk prefix from `prev_actions`.

In practice, clamping can happen either before denoise evaluation, after the Euler update, or both. The first version should clamp after every Euler update and before the next denoise step so the prefix remains fixed throughout the trajectory.

### Fallback behavior

If `rtc_config` is absent or the required prefix inputs are missing, fall back to normal `sample_actions()` behavior rather than crashing.

### Relationship to existing RTC modes

Keep these behaviors available and unchanged:

- `raw_paper`
- `executed_overlap_paper`

This preserves the current A/B baselines. The new training-time checkpoint can then be compared against:

- baseline checkpoint + `no_rtc`
- baseline checkpoint + `raw_paper`
- baseline checkpoint + `executed_overlap_paper`
- training-time RTC checkpoint + `training_time_prefix`

## Config and File Layout

### Model config

Extend the JAX model config to carry training-time RTC settings instead of hard-coding them inside `compute_loss()`.

Add a small config object, attached to `Pi0Config`, with fields like:

- `enabled: bool`
- `delay_values: tuple[int, ...]`
- `delay_probs: tuple[float, ...]`

The first version should default to disabled so existing training configs remain unchanged.

### Training config

Keep `pi05_agileX_thor_jax` unchanged and add a sibling train config for the new checkpoint:

- `pi05_agileX_thor_jax_training_time_rtc`

That config should reuse the same model/data setup but enable training-time RTC with the measured 30Hz delay sampler.

### Sampling config

Extend `PaperRTCConfig` mode handling to accept:

- `training_time_prefix`

This mode should bypass `apply_rtc_guidance()` entirely and instead use prefix clamping in `Pi0.sample_actions_rtc()`.

### Affected files

- `src/openpi/models/pi0_config.py`
  - add model-side training-time RTC config
- `src/openpi/models/pi0.py`
  - add training-time RTC loss path
  - add `training_time_prefix` sampling path
- `src/openpi/models/rtc_utils_jax.py`
  - extend mode resolution to include `training_time_prefix`
- `src/openpi/training/config.py`
  - add the new JAX training config
- `scripts/inference_jax_rtc_0325.py`
  - expose the new runtime mode for evaluation
- `src/openpi/models/model_test.py`
  - add model-level training and sampling tests
- `src/openpi/models/rtc_utils_jax_test.py`
  - add mode-resolution coverage for the new mode

## Testing Strategy

### Unit and model tests

Add focused tests that prove:

- delay sampling only emits configured values
- training-time RTC loss keeps the prefix clean and masks loss to the postfix
- `training_time_prefix` sampling runs under JIT and preserves output shape
- prefix positions remain clamped when `prev_actions` and `inference_delay` are provided
- existing `raw_paper` and `executed_overlap_paper` mode resolution stays intact

### Syntax and smoke checks

Run at minimum:

- `py_compile` on touched model/runtime files
- focused pytest for `src/openpi/models/model_test.py`
- focused pytest for `src/openpi/models/rtc_utils_jax_test.py`

### Robot-facing evaluation

First compare:

- baseline checkpoint with `no_rtc`
- baseline checkpoint with `raw_paper`
- baseline checkpoint with `executed_overlap_paper`
- training-time RTC checkpoint with `training_time_prefix`

Primary metrics:

- handoff jump at chunk boundaries
- overlap smoothness
- inference latency per chunk

Expected result:

- `training_time_prefix` should be much closer to `no_rtc` latency than the current guidance-based RTC modes
- handoff quality should improve relative to `no_rtc`

## Risks and Follow-Ups

### Main risk

The no-architecture-change version may underperform if global timestep conditioning is too weak to represent the prefix/postfix asymmetry.

### First fallback

If the checkpoint trains cleanly but gains are small, the next iteration should add explicit internal prefix/postfix masks to the model path without changing the outer robot runtime.

### Deferred work

Do not include these in v1:

- mixed 20Hz/30Hz delay sampling
- executed-space training loss
- gripper-specific overlap objectives
- per-token timestep conditioning
- PyTorch parity
