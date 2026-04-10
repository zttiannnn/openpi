# JAX Executed-Overlap RTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JAX `executed_overlap_paper` RTC mode that applies overlap guidance in executed joint space while keeping the current `raw_paper` JAX RTC path available as the baseline.

**Architecture:** Keep one JAX RTC entrypoint in `Pi0.sample_actions_rtc`, but extend the JAX RTC config so `apply_rtc_guidance` can switch between raw-space guidance and executed-space guidance. Executed-space guidance will decode denoised candidate actions through the same output-transform semantics that matter for handoff quality in AgileX: unnormalize, delta-to-absolute with current observation state, and joint flip. Runtime wiring in `scripts/inference_jax_rtc_0325.py` will pass both raw leftover and processed leftover so the model can optimize against the actual executed overlap target.

**Tech Stack:** Python, JAX, NumPy, Flax/NNX, existing RTC utilities in `src/openpi/models/rtc_utils_jax.py`, `src/openpi/models/pi0.py`, runtime wiring in `scripts/inference_jax_rtc_0325.py`, and existing test files `src/openpi/models/rtc_utils_jax_test.py` and `src/openpi/models/model_test.py`.

---

## Scope Locks

- First version adds one new JAX mode: `executed_overlap_paper`.
- Keep existing `raw_paper` behavior unchanged and selectable for A/B comparison.
- Executed-space loss only covers the first 6 arm joints.
- Do not include gripper guidance in the first version.
- Do not differentiate through hardware send operations such as `int()` or `abs()`.
- Decode only the transform stages that are already available in policy/runtime context: unnormalize, absolute-action reconstruction, and joint flip.

## File Map

- Modify: `src/openpi/models/rtc_utils_jax.py`
  - Extend RTC config and add executed-space decode/guidance helpers.
- Modify: `src/openpi/models/pi0.py`
  - Thread new RTC mode inputs into `sample_actions_rtc`.
- Modify: `src/openpi/policies/policy_config.py`
  - Build and attach a JAX-friendly executed transform spec, not just the PyTorch version.
- Modify: `scripts/inference_jax_rtc_0325.py`
  - Pass `processed_leftover`, `observation_state`, and mode information into `rtc_context`.
- Modify: `src/openpi/models/rtc_utils_jax_test.py`
  - Add utility tests for executed-space decoding and guidance.
- Modify: `src/openpi/models/model_test.py`
  - Add a focused model-level smoke test for `executed_overlap_paper`.
- Reference only: `src/openpi/models_pytorch/rtc_utils.py`
  - Use as the semantic reference for `decode_actions_to_executed_prefix_torch` and `executed_overlap_paper` behavior.

## Task 1: Define a shared JAX executed-transform spec

**Files:**
- Modify: `src/openpi/models/rtc_utils_jax.py`
- Modify: `src/openpi/policies/policy_config.py`

- [ ] **Step 1: Add a JAX-friendly executed transform spec dataclass**

Create a small dataclass in `src/openpi/models/rtc_utils_jax.py` with only the fields the JAX path needs:
- `action_mean`
- `action_std`
- `action_q01`
- `action_q99`
- `use_quantiles`
- `delta_action_mask`
- `output_joint_flip_mask`
- `arm_joint_dims`

Do not import the PyTorch dataclass into the JAX model path.

- [ ] **Step 2: Add a builder in `policy_config.py` for the JAX spec**

Mirror the existing logic of `_build_rtc_executed_prefix_transform_spec(...)`, but return the JAX dataclass when the loaded model is JAX. Reuse the same discovery rules:
- extract action stats from `norm_stats`
- detect `AbsoluteActions.mask`
- detect `AgileXOutputs(adapt_to_pi=True)` and fetch `_joint_flip_mask()`
- set `arm_joint_dims=6`

- [ ] **Step 3: Attach the spec to the created JAX model**

When `create_trained_policy(...)` builds a JAX model, attach the new spec as a model attribute, for example:
- `model.rtc_executed_prefix_transform_spec = ...`

Keep the existing PyTorch behavior unchanged.

- [ ] **Step 4: Verify attachment is non-breaking**

Run:
```bash
.venv/bin/python -m py_compile src/openpi/models/rtc_utils_jax.py src/openpi/policies/policy_config.py
```

Expected: no syntax errors.

## Task 2: Implement JAX executed-space decoding helpers

**Files:**
- Modify: `src/openpi/models/rtc_utils_jax.py`
- Reference: `src/openpi/models_pytorch/rtc_utils.py:1057`

- [ ] **Step 1: Add `decode_actions_to_executed_prefix_jax(...)`**

Implement a JAX version of the PyTorch helper with this contract:
- inputs:
  - `raw_actions`: `(B, H, D)` or `(H, D)`
  - `observation_state`: `(B, D)` or `(D,)`
  - `transform_spec`
- outputs:
  - decoded executed arm actions shaped `(B, H, arm_joint_dims)`

Decode order must be:
1. unnormalize or dequantile action outputs
2. if `delta_action_mask` exists, add masked observation-state prefix back in
3. if `output_joint_flip_mask` exists, multiply corresponding dims
4. slice to `arm_joint_dims`

- [ ] **Step 2: Add per-joint loss scaling helper**

Implement a JAX equivalent of `get_executed_prefix_loss_scale_torch(...)` so loss terms in executed space are normalized by joint scale rather than raw magnitude.

- [ ] **Step 3: Add shape and dtype normalization helpers**

Reuse or extend the existing batched-action helpers so the decoder:
- accepts batched/unbatched inputs
- preserves stable `float32` math for loss evaluation
- returns a predictable shape under `jit`

- [ ] **Step 4: Add focused decoder tests**

In `src/openpi/models/rtc_utils_jax_test.py`, add tests that verify:
- z-score unnormalize path
- quantile unnormalize path
- `delta_action_mask` adds state back correctly
- joint flip is applied correctly
- decoder returns the first 6 dims only when `arm_joint_dims=6`

- [ ] **Step 5: Run focused decoder tests**

Run:
```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py -k "decode or scale"
```

Expected: new decoder tests pass.

## Task 3: Extend the JAX RTC config and guidance switch

**Files:**
- Modify: `src/openpi/models/rtc_utils_jax.py`
- Modify: `src/openpi/models/pi0.py`

- [ ] **Step 1: Extend `PaperRTCConfig` into a multi-mode config**

Keep backward compatibility for existing launchers, but add fields such as:
- `mode: str = "raw_paper"`
- `beta: float = 5.0`
- `arm_joint_dims: int = 6`

Do not remove `enabled`.

- [ ] **Step 2: Add an executed-overlap guidance branch**

Inside `apply_rtc_guidance(...)`, keep the current raw branch intact and add a second branch:
- decode `x1_t = x_t - time * v_t` into executed joint space
- compare decoded overlap against `processed_leftover`
- build paper-style overlap weights using the same delay/execution-horizon mask logic
- compute weighted squared error in executed space
- backprop the executed-space loss to `x_t`
- convert this to the same guided-velocity form used by the raw path

- [ ] **Step 3: Define the minimal executed inputs required by the JAX guidance path**

`apply_rtc_guidance(...)` should accept new optional arguments:
- `processed_leftover`
- `observation_state`
- `executed_transform_spec`

If any of them are missing while `mode == "executed_overlap_paper"`, fall back to unguided or raw-mode behavior instead of crashing.

- [ ] **Step 4: Thread the new inputs through `Pi0.sample_actions_rtc(...)`**

Extend the JAX sample method signature with matching optional kwargs and forward them into `apply_rtc_guidance(...)`.

Keep the existing early fallback:
- if `rtc_config is None`
- or `prev_actions is None` in raw mode
then use normal sampling.

- [ ] **Step 5: Add a model-level smoke test**

In `src/openpi/models/model_test.py`, add one new test that calls `sample_actions_rtc(...)` with:
- `mode="executed_overlap_paper"`
- zero `processed_leftover`
- zero `observation_state`
- a simple transform spec

Assert only:
- output shape is correct
- the jitted function runs successfully

- [ ] **Step 6: Run focused JAX RTC tests**

Run:
```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py src/openpi/models/model_test.py -k "rtc"
```

Expected: tests pass with both raw and executed-overlap modes present.

## Task 4: Wire runtime context for executed-space RTC

**Files:**
- Modify: `scripts/inference_jax_rtc_0325.py`
- Reference: `src/openpi/shared/inference_kwargs.py`

- [ ] **Step 1: Add a runtime CLI mode switch**

Extend the launcher/runtime CLI with a small controlled choice set:
- `--rtc_mode raw_paper`
- `--rtc_mode executed_overlap_paper`

Default should remain `raw_paper` so existing behavior is preserved.

- [ ] **Step 2: Pass processed leftover into `rtc_context`**

When sending an async RTC request, the runtime already has access to both:
- `current_leftover = action_queue.get_raw_left_over()`
- `processed_leftover = action_queue.get_processed_left_over()`

Pass both into `rtc_context`, along with:
- `observation_state`
- `execution_horizon`
- `inference_delay`
- `rtc_config`

- [ ] **Step 3: Use the current observation state as the decode anchor**

At request time, use the same observation that is sent to the worker as the anchor state for executed decoding. Do not use an old queue head or stale state.

- [ ] **Step 4: Keep diagnostics split by raw and executed spaces**

Do not remove current handoff dumps. Continue logging:
- raw overlap stats
- executed overlap stats
- queue boundary jump

Add mode information so experiments can compare `raw_paper` vs `executed_overlap_paper` in the same plotting workflow.

- [ ] **Step 5: Run a syntax check on runtime wiring**

Run:
```bash
.venv/bin/python -m py_compile scripts/inference_jax_rtc_0325.py src/openpi/models/pi0.py src/openpi/models/rtc_utils_jax.py
```

Expected: no syntax errors.

## Task 5: Add regression-oriented tests and analysis hooks

**Files:**
- Modify: `src/openpi/models/rtc_utils_jax_test.py`
- Modify: `scripts/inference_jax_rtc_0325.py` if additional diagnostics are needed

- [ ] **Step 1: Add an executed-overlap improvement unit test**

Create a small deterministic test where:
- baseline decoded overlap has known error to a processed target
- executed-overlap guidance is applied once
- decoded overlap error decreases relative to the unguided step

This should be a utility-level test using a simple linear denoiser, not a full Pi0 model.

- [ ] **Step 2: Add a missing-input fallback test**

Verify that if mode is `executed_overlap_paper` but any of the following are missing:
- `processed_leftover`
- `observation_state`
- `executed_transform_spec`

the function returns the unguided baseline rather than raising under `jit`.

- [ ] **Step 3: Add a runtime diagnostics expectation list**

Document in the runtime log or plan comments that the first on-robot comparison should look at:
- `overlap_l2_after`
- `queue_boundary abs_max`
- executed chunk handoff PNGs
- `infer_ms`

The first success criterion is not “perfect smoothness”; it is “executed overlap metrics improve relative to raw mode without unacceptable latency regression.”

- [ ] **Step 4: Run the full targeted test set**

Run:
```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py src/openpi/models/model_test.py
```

Expected: all relevant JAX model and RTC tests pass.

## Task 6: On-robot validation protocol

**Files:**
- No code changes required unless issues are discovered

- [ ] **Step 1: Run raw baseline with current launcher**

Use the current launcher and save:
- terminal logs
- `.npz` handoff dumps
- rendered handoff plots

- [ ] **Step 2: Run executed-overlap mode with the same checkpoint/task**

Use the same:
- task prompt
- checkpoint
- fps
- num_steps
- robot side setup

Only switch `--rtc_mode`.

- [ ] **Step 3: Compare four metrics side by side**

Compare:
- median `real_delay`
- `RTC result overlap_l2_after`
- `queue_boundary abs_max`
- subjective handoff smoothness from executed plots / robot video

- [ ] **Step 4: Decide next branch based on outcome**

If executed-overlap improves handoff smoothness without major latency regression, keep it and tune beta/mask settings.

If it does not, stop before adding more heuristics and reassess whether the dominant issue is:
- state anchoring
- gripper behavior
- send-layer quantization
- control-frequency mismatch

## Risks and non-goals

- This plan does not solve gripper discontinuity in the first pass.
- This plan does not make the hardware send path differentiable.
- This plan may increase JAX compile time and per-step RTC latency.
- The first version should not attempt to replicate the PyTorch `executed_prefix_b1` branch; that is a separate follow-up if executed-overlap proves promising.
