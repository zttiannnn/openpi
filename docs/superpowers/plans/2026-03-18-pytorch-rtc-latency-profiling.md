# PyTorch RTC Latency Profiling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in profiling path that reports comparable latency breakdowns for vanilla pi0.5 inference and RTC/B1 inference.

**Architecture:** Introduce a lightweight manual profiler shared by `sample_actions()` and `sample_actions_rtc()`, thread the profiling payload through `Policy.infer()`, and expose single-trace plus aggregate summaries via the inference script.

**Tech Stack:** Python, PyTorch, NumPy, existing `openpi` inference CLI.

---

## Chunk 1: Profiling Data Model

### Task 1: Add failing tests for profiler helpers

**Files:**
- Modify: `openpi/src/openpi/models_pytorch/rtc_utils_test.py`
- Modify: `openpi/src/openpi/models_pytorch/rtc_utils.py`

- [ ] **Step 1: Write failing tests**

Add tests for:
- recording named timing events
- aggregating single-trace and multi-run summaries
- formatting a paper-like summary row set

- [ ] **Step 2: Run tests to verify failure**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: FAIL because profiling helpers do not exist yet.

- [ ] **Step 3: Implement minimal profiler helpers**

Add focused helpers in `rtc_utils.py`:
- event/timer recorder
- summary aggregation
- summary formatting

- [ ] **Step 4: Run tests to verify pass**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: PASS for the new tests.

## Chunk 2: Model Instrumentation

### Task 2: Instrument no-RTC sampling

**Files:**
- Modify: `openpi/src/openpi/models_pytorch/pi0_pytorch.py`
- Modify: `openpi/src/openpi/policies/policy.py`
- Test: `openpi/src/openpi/models_pytorch/rtc_utils_test.py`

- [ ] **Step 1: Write failing tests**

Add tests that the output payload can contain:
- `image_encoders_ms`
- `llm_prefill_ms`
- `denoising_total_ms`
- `denoising_step_mean_ms`
- `total_model_ms`

- [ ] **Step 2: Run tests to verify failure**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: FAIL because payload fields are absent.

- [ ] **Step 3: Implement no-RTC instrumentation**

Instrument:
- image encode region in `embed_prefix()`
- prefill forward in `sample_actions()`
- each denoise step
- end-to-end model call

Thread the payload through `Policy.infer()`.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: PASS for new output payload tests.

### Task 3: Instrument RTC/B1 guidance

**Files:**
- Modify: `openpi/src/openpi/models_pytorch/pi0_pytorch.py`
- Modify: `openpi/src/openpi/models_pytorch/rtc_utils.py`
- Test: `openpi/src/openpi/models_pytorch/rtc_utils_test.py`

- [ ] **Step 1: Write failing tests**

Add tests for RTC-specific summary fields:
- `rtc_guidance_total_ms`
- `rtc_decode_ms`
- `rtc_loss_ms`
- `rtc_autograd_ms`
- `rtc_diag_ms`

- [ ] **Step 2: Run tests to verify failure**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: FAIL because RTC timing fields are absent.

- [ ] **Step 3: Implement RTC instrumentation**

Instrument `apply_rtc_guidance()` and `sample_actions_rtc()` so RTC timings are recorded with the same profiler data model.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: PASS.

## Chunk 3: CLI and Reporting

### Task 4: Add profiling CLI flags and output files

**Files:**
- Modify: `openpi/scripts/inference_smooth_0912.py`
- Modify: `openpi/src/openpi/policies/policy.py`
- Modify: `openpi/src/openpi/shared/inference_kwargs.py`

- [ ] **Step 1: Write failing tests or lightweight assertions**

Add coverage for CLI-facing summary formatting helpers in `rtc_utils_test.py`.

- [ ] **Step 2: Run tests to verify failure**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: FAIL because CLI summary formatting/output helpers are incomplete.

- [ ] **Step 3: Implement CLI support**

Add flags such as:
- `--profile_model_enable`
- `--profile_model_warmup_runs`
- `--profile_model_runs`
- `--profile_model_output_json`
- `--profile_model_output_csv`

Print:
- single-trace table
- aggregate mean/std table

Write:
- JSON trace file
- CSV summary file

- [ ] **Step 4: Run tests to verify pass**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: PASS.

## Chunk 4: Verification

### Task 5: Verify no-RTC and RTC profiling end-to-end

**Files:**
- Modify: `openpi/scripts/debug/` scripts if needed

- [ ] **Step 1: Run unit tests**

Run: `cd openpi && python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py`
Expected: PASS.

- [ ] **Step 2: Run syntax verification**

Run: `cd openpi && python -m py_compile src/openpi/models_pytorch/pi0_pytorch.py src/openpi/models_pytorch/rtc_utils.py src/openpi/policies/policy.py scripts/inference_smooth_0912.py src/openpi/shared/inference_kwargs.py`
Expected: PASS.

- [ ] **Step 3: Smoke-test no-RTC profiling**

Run one inference command with profiling enabled and RTC disabled.
Expected: terminal table plus JSON/CSV outputs with populated no-RTC fields.

- [ ] **Step 4: Smoke-test RTC profiling**

Run one inference command with profiling enabled and RTC enabled.
Expected: terminal table plus JSON/CSV outputs with populated RTC-specific fields.

- [ ] **Step 5: Summarize comparison**

Confirm we can directly compare:
- image encoder time
- prefill time
- denoising total and per-step time
- RTC guidance internal breakdown
