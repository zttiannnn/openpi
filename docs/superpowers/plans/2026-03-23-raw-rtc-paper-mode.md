# Raw RTC Paper Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-style raw RTC mode while keeping the existing `raw_prefix` implementation unchanged as a legacy baseline.

**Architecture:** The model-side RTC path stays unified in `apply_rtc_guidance`, but raw guidance mask construction is split into legacy and paper variants. CLI wiring exposes the new mode and an `auto` schedule option so experiments can switch between engineering and paper-faithful raw RTC without changing code.

**Tech Stack:** Python, PyTorch, NumPy, existing RTC utilities and tests in `src/openpi/models_pytorch/rtc_utils.py`, `src/openpi/models_pytorch/rtc_utils_test.py`, and `scripts/inference_smooth_0912.py`

---

## File Map

- Modify: `src/openpi/models_pytorch/rtc_utils.py`
  - Add paper-style raw RTC mask selection and mode handling
- Modify: `src/openpi/models_pytorch/rtc_utils_test.py`
  - Add failing tests for new raw RTC mode semantics
- Modify: `scripts/inference_smooth_0912.py`
  - Expose the new mode and `auto` schedule in the CLI

## Chunk 1: Tests for New Mode Semantics

### Task 1: Add failing tests for `raw_prefix_paper`

**Files:**
- Modify: `src/openpi/models_pytorch/rtc_utils_test.py`

- [ ] **Step 1: Write failing config/default tests**

Add tests that verify:
- `RTCConfig()` still defaults to legacy `raw_prefix`
- `raw_prefix_paper` is accepted explicitly
- schedule resolution can distinguish legacy from paper mode

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:
```bash
PYTHONPATH=/home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/src:$PYTHONPATH pytest -q src/openpi/models_pytorch/rtc_utils_test.py -k "paper or schedule"
```

Expected: FAIL because the new mode/schedule resolution does not exist yet.

- [ ] **Step 3: Write failing raw guidance behavior tests**

Add one focused behavior test showing that:
- legacy `raw_prefix` remains truncated by `execution_horizon`
- `raw_prefix_paper` keeps guiding over the full overlap region for the same input

- [ ] **Step 4: Run the focused tests and verify they fail**

Run the same command as above.

Expected: FAIL because `raw_prefix_paper` behavior is not implemented yet.

## Chunk 2: Minimal Runtime Implementation

### Task 2: Implement the new raw RTC mode in `rtc_utils.py`

**Files:**
- Modify: `src/openpi/models_pytorch/rtc_utils.py`

- [ ] **Step 1: Add schedule resolution helper**

Implement a helper that maps:
- `auto` + `raw_prefix` -> `linear`
- `auto` + `raw_prefix_paper` -> `exp`
- explicit values -> unchanged

- [ ] **Step 2: Add raw mask-end selection helper**

Implement a helper that returns:
- legacy raw mode: `execution_horizon`
- paper raw mode: overlap-aware end derived from available leftover length

- [ ] **Step 3: Wire the helper into `apply_rtc_guidance`**

Only raw guidance behavior should change.

- [ ] **Step 4: Run focused tests and verify they pass**

Run:
```bash
PYTHONPATH=/home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/src:$PYTHONPATH pytest -q src/openpi/models_pytorch/rtc_utils_test.py -k "paper or schedule"
```

Expected: PASS

## Chunk 3: CLI Wiring

### Task 3: Expose the new mode in the debug/inference entrypoint

**Files:**
- Modify: `scripts/inference_smooth_0912.py`

- [ ] **Step 1: Expand CLI choices**

Add:
- `raw_prefix_paper` to `--rtc_guidance_mode`
- `auto` to `--rtc_schedule`

- [ ] **Step 2: Keep legacy behavior stable**

Default behavior should remain unchanged for existing scripts.

- [ ] **Step 3: Run a syntax check**

Run:
```bash
PYTHONPATH=/home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python - <<'PY'
import ast
from pathlib import Path
for path in [
    Path('scripts/inference_smooth_0912.py'),
    Path('src/openpi/models_pytorch/rtc_utils.py'),
    Path('src/openpi/models_pytorch/rtc_utils_test.py'),
]:
    ast.parse(path.read_text(), filename=str(path))
    print(f'AST OK: {path}')
PY
```

Expected: all files parse successfully.

## Chunk 4: Targeted Verification

### Task 4: Verify legacy stability and new mode entrypoint

**Files:**
- Modify if needed: `src/openpi/models_pytorch/rtc_utils.py`
- Modify if needed: `src/openpi/models_pytorch/rtc_utils_test.py`
- Modify if needed: `scripts/inference_smooth_0912.py`

- [ ] **Step 1: Run targeted RTC utility tests**

Run:
```bash
PYTHONPATH=/home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/src:$PYTHONPATH pytest -q src/openpi/models_pytorch/rtc_utils_test.py
```

Expected: PASS, or a pre-existing environment/import blocker outside this change.

- [ ] **Step 2: Summarize new usage**

Document the new experimental invocation, for example:
```bash
python scripts/inference_smooth_0912.py ... \
  --rtc_enable \
  --rtc_guidance_mode raw_prefix_paper \
  --rtc_schedule auto
```
