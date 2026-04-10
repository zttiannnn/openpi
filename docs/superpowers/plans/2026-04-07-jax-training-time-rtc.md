# JAX Training-Time RTC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a JAX training-time RTC checkpoint and a matching `training_time_prefix` sampling mode for `pi05_agileX_thor_jax`, while preserving the current JAX inference-time RTC baselines.

**Architecture:** Keep the existing `Pi0.5` network unchanged and move RTC logic into two places: training-time chunk construction in `Pi0.compute_loss()` and prefix clamping in `Pi0.sample_actions_rtc()`. Reuse the current async RTC runtime interface so the new checkpoint can be evaluated by swapping `rtc_mode` without introducing a new robot-side protocol.

**Tech Stack:** Python, JAX, Flax NNX, Tyro train configs, existing JAX RTC helpers in `src/openpi/models/rtc_utils_jax.py`, model integration in `src/openpi/models/pi0.py`, runtime wiring in `scripts/inference_jax_rtc_0325.py`, and focused pytest coverage in `src/openpi/models/rtc_utils_jax_test.py` and `src/openpi/models/model_test.py`.

---

## File Map

- `src/openpi/models/pi0_config.py`
  - Add the frozen model-side config object that turns training-time RTC on and stores the measured 30Hz delay distribution.
- `src/openpi/models/rtc_utils_jax.py`
  - Extend mode resolution and add reusable helpers for sampling delay steps, building training-time prefix/postfix masks, and clamping a prefix during sampling.
- `src/openpi/models/pi0.py`
  - Integrate the training-time RTC loss path into `compute_loss()` and add the new `training_time_prefix` sampling branch inside `sample_actions_rtc()`.
- `src/openpi/training/config.py`
  - Add the new train config `pi05_agileX_thor_jax_training_time_rtc`.
- `scripts/inference_jax_rtc_0325.py`
  - Expose `training_time_prefix` on the CLI and keep runtime diagnostics working for the new mode.
- `src/openpi/models/rtc_utils_jax_test.py`
  - Add helper-level tests for delay sampling, prefix/postfix construction, prefix clamping, and mode resolution.
- `src/openpi/models/model_test.py`
  - Add smoke tests for the enabled loss path, the new train config, the sampling-time prefix clamp, and runtime context behavior.

### Task 1: Add Training-Time RTC Config And JAX Helpers

**Files:**
- Modify: `src/openpi/models/pi0_config.py:18-110`
- Modify: `src/openpi/models/rtc_utils_jax.py:22-120`
- Modify: `src/openpi/models/rtc_utils_jax_test.py:1-260`

- [ ] **Step 1: Write the failing helper/config tests**

Add these tests near the top of `src/openpi/models/rtc_utils_jax_test.py`:

```python
def test_resolve_rtc_mode_accepts_training_time_prefix():
    cfg = rtc_utils_jax.PaperRTCConfig(enabled=True, mode="training_time_prefix")
    assert rtc_utils_jax.resolve_rtc_mode(cfg) == "training_time_prefix"


def test_sample_training_time_delay_steps_emits_only_configured_values():
    delays = rtc_utils_jax.sample_training_time_delay_steps(
        jax.random.key(0),
        delay_values=(4, 5),
        delay_probs=(0.8, 0.2),
        batch_shape=(512,),
    )

    assert delays.shape == (512,)
    assert set(np.asarray(jnp.unique(delays)).tolist()) <= {4, 5}


def test_build_training_time_rtc_inputs_keeps_prefix_clean_and_masks_postfix():
    actions = jnp.arange(12, dtype=jnp.float32).reshape(1, 4, 3)
    noise = jnp.full((1, 4, 3), 100.0, dtype=jnp.float32)
    time = jnp.array([0.5], dtype=jnp.float32)

    x_t, loss_weight = rtc_utils_jax.build_training_time_rtc_inputs(
        actions=actions,
        noise=noise,
        time=time,
        delay_steps=jnp.array([2], dtype=jnp.int32),
    )

    expected_noisy = 0.5 * noise + 0.5 * actions
    np.testing.assert_allclose(np.asarray(x_t[:, :2]), np.asarray(actions[:, :2]), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(np.asarray(x_t[:, 2:]), np.asarray(expected_noisy[:, 2:]), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(np.asarray(loss_weight), np.asarray([[0.0, 0.0, 2.0, 2.0]]), atol=1e-6, rtol=0.0)


def test_clamp_prefix_actions_replaces_only_the_clipped_prefix():
    actions = jnp.zeros((1, 4, 2), dtype=jnp.float32)
    prev_actions = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)

    clamped = rtc_utils_jax.clamp_prefix_actions(actions, prev_actions, prefix_steps=2)
    fully_clamped = rtc_utils_jax.clamp_prefix_actions(actions, prev_actions, prefix_steps=10)

    np.testing.assert_allclose(np.asarray(clamped[:, :2]), np.asarray(prev_actions[:, :2]), atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(np.asarray(clamped[:, 2:]), 0.0, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(np.asarray(fully_clamped), np.asarray(prev_actions), atol=1e-6, rtol=0.0)
```

- [ ] **Step 2: Run the focused tests and confirm they fail for missing helpers**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py -k "training_time or clamp_prefix or resolve_rtc_mode_accepts"
```

Expected: FAIL with missing attributes such as `sample_training_time_delay_steps`, `build_training_time_rtc_inputs`, or `training_time_prefix` not being accepted by `resolve_rtc_mode`.

- [ ] **Step 3: Add the frozen model config and new RTC helper functions**

In `src/openpi/models/pi0_config.py`, add the config surface:

```python
@dataclasses.dataclass(frozen=True)
class TrainingTimeRTCConfig:
    enabled: bool = False
    delay_values: tuple[int, ...] = (4, 5)
    delay_probs: tuple[float, ...] = (188 / 199, 11 / 199)

    def __post_init__(self):
        if len(self.delay_values) == 0:
            raise ValueError("delay_values must not be empty")
        if len(self.delay_values) != len(self.delay_probs):
            raise ValueError("delay_values and delay_probs must have the same length")
        if any(step < 0 for step in self.delay_values):
            raise ValueError("delay_values must be non-negative")
        if any(prob <= 0.0 for prob in self.delay_probs):
            raise ValueError("delay_probs must be positive")
```

Then attach it to `Pi0Config`:

```python
training_time_rtc: TrainingTimeRTCConfig = dataclasses.field(default_factory=TrainingTimeRTCConfig)
```

In `src/openpi/models/rtc_utils_jax.py`, extend mode resolution and add the reusable helpers:

```python
def resolve_rtc_mode(rtc_config: PaperRTCConfig) -> str:
    mode = rtc_config.guidance_mode if rtc_config.guidance_mode is not None else rtc_config.mode
    if mode in {"raw_paper", "raw_prefix_paper", "raw_prefix"}:
        return "raw_paper"
    if mode in {"executed_overlap_paper", "training_time_prefix"}:
        return mode
    raise ValueError(f"Unknown RTC mode: {mode}")


def sample_training_time_delay_steps(
    rng: jax.Array,
    *,
    delay_values: tuple[int, ...],
    delay_probs: tuple[float, ...],
    batch_shape: tuple[int, ...],
) -> jax.Array:
    values = jnp.asarray(delay_values, dtype=jnp.int32)
    probs = jnp.asarray(delay_probs, dtype=jnp.float32)
    probs = probs / jnp.sum(probs)
    logits = jnp.log(jnp.maximum(probs, 1e-8))
    indices = jax.random.categorical(rng, logits, shape=batch_shape)
    return values[indices]


def make_prefix_mask(delay_steps: int | jax.Array, *, action_horizon: int) -> jax.Array:
    delay_steps = jnp.clip(jnp.asarray(delay_steps, dtype=jnp.int32), 0, action_horizon)
    return jnp.arange(action_horizon, dtype=jnp.int32) < delay_steps[..., None]


def build_training_time_rtc_inputs(
    *,
    actions: jax.Array,
    noise: jax.Array,
    time: jax.Array,
    delay_steps: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    prefix_mask = make_prefix_mask(delay_steps, action_horizon=actions.shape[-2])
    time_expanded = jnp.asarray(time, dtype=actions.dtype)[..., None, None]
    noisy_actions = time_expanded * noise + (1.0 - time_expanded) * actions
    x_t = jnp.where(prefix_mask[..., None], actions, noisy_actions)
    postfix_mask = (~prefix_mask).astype(actions.dtype)
    postfix_count = jnp.maximum(jnp.sum(postfix_mask, axis=-1, keepdims=True), 1.0)
    loss_weight = postfix_mask * (actions.shape[-2] / postfix_count)
    return x_t, loss_weight


def clamp_prefix_actions(
    actions: jax.Array,
    prev_actions: jax.Array | None,
    prefix_steps: int | jax.Array,
) -> jax.Array:
    actions, squeezed = _ensure_batched_actions(jnp.asarray(actions))
    if prev_actions is None:
        return actions[0] if squeezed else actions

    prev_actions = _pad_prev_actions(
        prev_actions,
        batch_size=actions.shape[0],
        action_horizon=actions.shape[1],
        action_dim=actions.shape[2],
        dtype=actions.dtype,
    )
    prefix_mask = make_prefix_mask(prefix_steps, action_horizon=actions.shape[1])
    prefix_mask = jnp.broadcast_to(prefix_mask, actions.shape[:2])
    clamped = jnp.where(prefix_mask[..., None], prev_actions, actions)
    return clamped[0] if squeezed else clamped
```

- [ ] **Step 4: Run the helper tests and a syntax check**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py -k "training_time or clamp_prefix or resolve_rtc_mode_accepts"
.venv/bin/python -m py_compile src/openpi/models/pi0_config.py src/openpi/models/rtc_utils_jax.py
```

Expected: the focused helper tests PASS and `py_compile` exits cleanly.

- [ ] **Step 5: Commit the helper/config foundation**

Run:

```bash
git add src/openpi/models/pi0_config.py src/openpi/models/rtc_utils_jax.py src/openpi/models/rtc_utils_jax_test.py
git commit -m "feat: add jax training-time rtc helpers"
```

### Task 2: Integrate The Training-Time RTC Loss And Train Config

**Files:**
- Modify: `src/openpi/models/pi0.py:66-90`
- Modify: `src/openpi/models/pi0.py:190-216`
- Modify: `src/openpi/training/config.py:1013-1049`
- Modify: `src/openpi/models/model_test.py:22-240`

- [ ] **Step 1: Write the failing loss/config tests**

Add these tests to `src/openpi/models/model_test.py`:

```python
def test_pi0_model_compute_loss_training_time_rtc_smoke():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        training_time_rtc=pi0_config.TrainingTimeRTCConfig(
            enabled=True,
            delay_values=(2,),
            delay_probs=(1.0,),
        )
    )
    model = config.create(key)
    obs, act = config.fake_obs(2), config.fake_act(2)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)

    assert loss.shape == (2, config.action_horizon)
    assert jnp.all(jnp.isfinite(loss))


def test_training_time_rtc_train_config_uses_measured_delay_distribution():
    config = _config.get_config("pi05_agileX_thor_jax_training_time_rtc")

    assert config.model.training_time_rtc.enabled is True
    assert config.model.training_time_rtc.delay_values == (4, 5)
    assert config.model.training_time_rtc.delay_probs == pytest.approx((188 / 199, 11 / 199))
```

- [ ] **Step 2: Run the focused tests and confirm they fail before the integration**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/model_test.py -k "training_time_rtc_smoke or measured_delay_distribution"
```

Expected: FAIL because the new train config does not exist yet and `compute_loss()` is not exercising the training-time RTC path.

- [ ] **Step 3: Implement the loss-path integration and add the sibling train config**

In `src/openpi/models/pi0.py`, keep the existing baseline path but branch when `self.training_time_rtc.enabled` is true:

```python
self.training_time_rtc = config.training_time_rtc
```

Then replace the top of `compute_loss()` with:

```python
preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

batch_shape = actions.shape[:-2]
noise = jax.random.normal(noise_rng, actions.shape)
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
u_t = noise - actions

if self.training_time_rtc.enabled:
    delay_steps = rtc_utils_jax.sample_training_time_delay_steps(
        delay_rng,
        delay_values=self.training_time_rtc.delay_values,
        delay_probs=self.training_time_rtc.delay_probs,
        batch_shape=batch_shape,
    )
    x_t, loss_weight = rtc_utils_jax.build_training_time_rtc_inputs(
        actions=actions,
        noise=noise,
        time=time,
        delay_steps=delay_steps,
    )
else:
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    loss_weight = jnp.ones(actions.shape[:-1], dtype=actions.dtype)
```

Then replace the return statement with:

```python
per_step_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
return per_step_loss * loss_weight
```

In `src/openpi/training/config.py`, add the sibling config immediately after `pi05_agileX_thor_jax`:

```python
TrainConfig(
    name="pi05_agileX_thor_jax_training_time_rtc",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=7,
        action_horizon=50,
        max_token_len=128,
        pi05=True,
        training_time_rtc=pi0_config.TrainingTimeRTCConfig(
            enabled=True,
            delay_values=(4, 5),
            delay_probs=(188 / 199, 11 / 199),
        ),
    ),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=7,
        action_horizon=50,
        max_token_len=128,
        pi05=True,
        training_time_rtc=pi0_config.TrainingTimeRTCConfig(
            enabled=True,
            delay_values=(4, 5),
            delay_probs=(188 / 199, 11 / 199),
        ),
    ).get_freeze_filter(),
    weight_loader=weight_loaders.CheckpointWeightLoader("/jedata/pi0_base/pi05_base/params"),
    data=LeRobotAgileXDataConfigThor(
        assets=AssetsConfig(assets_dir="/workspace/JE_robot_data_lerobot/0311_data_lerobot"),
        data_root="/workspace/JE_robot_data_lerobot/0311_data_lerobot",
        default_prompt="Put the purple carton of milk into the cardboard box.",
    ),
    policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    wandb_enabled=True,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2e-4,
        decay_steps=50_000,
        decay_lr=1e-6,
    ),
    num_train_steps=40_000,
    batch_size=512,
    log_interval=100,
    save_interval=2_500,
    keep_period=5_000,
    num_workers=8,
    fsdp_devices=8,
),
```

- [ ] **Step 4: Run the focused model/config tests and a syntax check**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/model_test.py -k "training_time_rtc_smoke or measured_delay_distribution"
.venv/bin/python -m py_compile src/openpi/models/pi0.py src/openpi/training/config.py
```

Expected: the new tests PASS and the modified files compile cleanly.

- [ ] **Step 5: Commit the training-loss integration**

Run:

```bash
git add src/openpi/models/pi0.py src/openpi/training/config.py src/openpi/models/model_test.py
git commit -m "feat: train pi0 with prefix-frozen rtc loss"
```

### Task 3: Add The `training_time_prefix` Sampling Mode And Runtime Wiring

**Files:**
- Modify: `src/openpi/models/pi0.py:288-353`
- Modify: `scripts/inference_jax_rtc_0325.py:216-565`
- Modify: `src/openpi/models/model_test.py:52-240`

- [ ] **Step 1: Write the failing sampling/runtime tests**

Add these tests to `src/openpi/models/model_test.py`:

```python
import numpy as np


def test_pi0_model_rtc_training_time_prefix_clamps_prefix():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)
    obs, _act = config.fake_obs(1), config.fake_act(1)

    prev_actions = jnp.full((1, model.action_horizon, model.action_dim), 7.0, dtype=jnp.float32)
    noise = jnp.linspace(
        -1.0,
        1.0,
        model.action_horizon * model.action_dim,
        dtype=jnp.float32,
    ).reshape(1, model.action_horizon, model.action_dim)

    actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=3,
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=True, mode="training_time_prefix", beta=5.0),
        prev_actions=prev_actions,
        inference_delay=4,
        execution_horizon=25,
        noise=noise,
    )

    np.testing.assert_allclose(np.asarray(actions[:, :4]), np.asarray(prev_actions[:, :4]), atol=1e-6, rtol=0.0)


def test_pi0_model_rtc_training_time_prefix_missing_prev_actions_matches_baseline():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)
    obs, _act = config.fake_obs(1), config.fake_act(1)
    noise = jnp.ones((1, model.action_horizon, model.action_dim), dtype=jnp.float32)

    baseline = nnx_utils.module_jit(model.sample_actions)(
        key,
        obs,
        num_steps=3,
        noise=noise,
    )
    rtc_actions = nnx_utils.module_jit(model.sample_actions_rtc)(
        key,
        obs,
        num_steps=3,
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=True, mode="training_time_prefix", beta=5.0),
        prev_actions=None,
        inference_delay=4,
        execution_horizon=25,
        noise=noise,
    )

    assert jnp.allclose(rtc_actions, baseline)


def test_build_rtc_context_training_time_prefix_skips_processed_leftover():
    rtc_context = _inference_jax_rtc_0325.build_rtc_context(
        rtc_config=rtc_utils_jax.PaperRTCConfig(enabled=True, mode="training_time_prefix", beta=5.0),
        prev_actions=jnp.zeros((2, 3), dtype=jnp.float32),
        processed_leftover=jnp.ones((2, 3), dtype=jnp.float32),
        processed_leftover_len=2,
        inference_delay=0,
        execution_horizon=25,
    )

    assert "processed_leftover" not in rtc_context


def test_resolve_runtime_guidance_mode_maps_training_time_prefix_to_raw_prefix_paper():
    assert _inference_jax_rtc_0325.resolve_runtime_guidance_mode("training_time_prefix") == "raw_prefix_paper"
```

- [ ] **Step 2: Run the focused sampling/runtime tests and confirm they fail first**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/model_test.py -k "training_time_prefix"
```

Expected: FAIL because `training_time_prefix` is not yet handled by `sample_actions_rtc()` or the runtime CLI helpers.

- [ ] **Step 3: Implement prefix clamping in `sample_actions_rtc()` and expose the runtime mode**

In `src/openpi/models/pi0.py`, branch early for the new mode:

```python
rtc_mode = rtc_utils_jax.resolve_rtc_mode(rtc_config)
if rtc_mode in {"raw_paper", "training_time_prefix"} and prev_actions is None:
    return self.sample_actions(rng, observation, num_steps=num_steps, noise=noise)
```

Clamp the initial noise when the mode is active:

```python
if rtc_mode == "training_time_prefix":
    noise = rtc_utils_jax.clamp_prefix_actions(noise, prev_actions, inference_delay)
```

Then split the step function:

```python
def step(carry):
    x_t, time = carry

    def denoise_step_partial(x_t_partial):
        return self._denoise_step_from_cache(
            observation,
            prefix_tokens,
            prefix_mask,
            kv_cache,
            x_t_partial,
            time,
        )

    if rtc_mode == "training_time_prefix":
        x_t = rtc_utils_jax.clamp_prefix_actions(x_t, prev_actions, inference_delay)
        v_t = denoise_step_partial(x_t)
        x_next = x_t + dt * v_t
        x_next = rtc_utils_jax.clamp_prefix_actions(x_next, prev_actions, inference_delay)
        return x_next, time + dt

    v_t = rtc_utils_jax.apply_rtc_guidance(
        x_t=x_t,
        prev_chunk_left_over=prev_actions,
        inference_delay=inference_delay,
        execution_horizon=execution_horizon,
        time=time,
        denoise_step_partial=denoise_step_partial,
        rtc_config=rtc_config,
        processed_leftover=processed_leftover,
        processed_leftover_len=processed_leftover_len,
        observation_state=observation_state,
        executed_transform_spec=executed_transform_spec,
    )
    return x_t + dt * v_t, time + dt
```

In `scripts/inference_jax_rtc_0325.py`, add the mode to the CLI and keep diagnostics compatible:

```python
parser.add_argument(
    "--rtc_mode",
    type=str,
    default="raw_paper",
    choices=("raw_paper", "executed_overlap_paper", "training_time_prefix"),
    help="RTC guidance mode",
)


def resolve_runtime_guidance_mode(rtc_mode: str) -> str:
    if rtc_mode in {"raw_paper", "training_time_prefix"}:
        return "raw_prefix_paper"
    return rtc_mode
```

Do not add `processed_leftover` to `build_rtc_context()` for `training_time_prefix`; leave that function’s current executed-only condition intact.

- [ ] **Step 4: Run focused tests plus syntax checks for the sampling/runtime path**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/model_test.py -k "training_time_prefix"
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py src/openpi/models/model_test.py -k "rtc"
.venv/bin/python -m py_compile src/openpi/models/pi0.py scripts/inference_jax_rtc_0325.py
```

Expected: the `training_time_prefix` tests PASS, the broader RTC subset still passes, and both files compile.

- [ ] **Step 5: Commit the new sampling/runtime mode**

Run:

```bash
git add src/openpi/models/pi0.py scripts/inference_jax_rtc_0325.py src/openpi/models/model_test.py
git commit -m "feat: add jax training-time rtc sampling mode"
```

### Task 4: Run End-To-End Focused Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run the helper and model test suite for all touched RTC paths**

Run:

```bash
.venv/bin/python -m pytest -q src/openpi/models/rtc_utils_jax_test.py src/openpi/models/model_test.py -k "rtc or training_time"
```

Expected: PASS with the new helper tests, model smoke tests, and runtime-context tests all green.

- [ ] **Step 2: Run syntax checks on every touched implementation file**

Run:

```bash
.venv/bin/python -m py_compile \
  src/openpi/models/pi0_config.py \
  src/openpi/models/rtc_utils_jax.py \
  src/openpi/models/pi0.py \
  src/openpi/training/config.py \
  scripts/inference_jax_rtc_0325.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Smoke-check the new training config can be resolved**

Run:

```bash
.venv/bin/python - <<'PY'
from openpi.training import config as _config

cfg = _config.get_config("pi05_agileX_thor_jax_training_time_rtc")
print(cfg.name)
print(cfg.model.training_time_rtc.enabled)
print(cfg.model.training_time_rtc.delay_values)
print(cfg.model.training_time_rtc.delay_probs)
PY
```

Expected:

```text
pi05_agileX_thor_jax_training_time_rtc
True
(4, 5)
(0.9447236180904522, 0.05527638190954774)
```
