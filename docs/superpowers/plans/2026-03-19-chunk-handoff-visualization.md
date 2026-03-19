# Chunk Handoff Visualization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a low-overhead chunk handoff dump path plus an offline plotting script so we can visualize prior/new action chunk disagreement around the handoff boundary.

**Architecture:** Runtime inference only saves compact `.npz` handoff snapshots at merge points; it does not render plots. A separate plotting script reads those snapshots and produces prior/new joint-space figures for both raw and executed spaces. Filtering is handled at dump time so experiments can save all handoffs or only selected step ranges.

**Tech Stack:** Python, NumPy, Matplotlib, existing RTC/debug utilities in `scripts/inference_smooth_0912.py` and `src/openpi/models_pytorch/rtc_utils.py`

---

## File Map

- Modify: `scripts/inference_smooth_0912.py`
  - Add chunk handoff dump CLI flags
  - Capture prior/new handoff arrays and save `.npz` snapshots
- Modify: `src/openpi/models_pytorch/rtc_utils.py`
  - Add helper(s) for handoff payload construction and step filtering
- Test: `src/openpi/models_pytorch/rtc_utils_test.py`
  - Add tests for filter logic and payload normalization
- Create: `scripts/plot_chunk_handoff.py`
  - Offline plotting script for `.npz` snapshots

## Chunk 1: Runtime Dump Helpers

### Task 1: Add failing tests for dump filtering and payload normalization

**Files:**
- Modify: `src/openpi/models_pytorch/rtc_utils_test.py`
- Test: `src/openpi/models_pytorch/rtc_utils_test.py`

- [ ] **Step 1: Write failing tests for step-range and stride filtering**

Add tests covering:
- dump disabled
- `step_min` / `step_max`
- `stride`
- combined filters

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
docker exec openpi_0207_v3 bash -lc 'cd /workspace/robot_repo/openpi_torch/openpi && PYTHONPATH=/workspace/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py -k "handoff_dump"'
```

Expected: FAIL because helper(s) do not exist yet.

- [ ] **Step 3: Write failing tests for payload normalization**

Add tests covering:
- `(D,)` input normalized to `(1, D)`
- `(T, D)` preserved
- `None` handled for optional arrays if allowed by helper contract

- [ ] **Step 4: Run test to verify it fails**

Run the same command as above.

Expected: FAIL on missing payload helper behavior.

- [ ] **Step 5: Implement minimal helpers in `rtc_utils.py`**

Add focused helpers, for example:
- `should_dump_chunk_handoff(step_idx, enabled, step_min, step_max, stride)`
- `build_chunk_handoff_dump_payload(...)`

Keep them NumPy-only and side-effect-free.

- [ ] **Step 6: Run tests to verify they pass**

Run:
```bash
docker exec openpi_0207_v3 bash -lc 'cd /workspace/robot_repo/openpi_torch/openpi && PYTHONPATH=/workspace/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py -k "handoff_dump"'
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi add src/openpi/models_pytorch/rtc_utils.py src/openpi/models_pytorch/rtc_utils_test.py
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi commit -m "feat: add chunk handoff dump helpers"
```

## Chunk 2: Runtime `.npz` Dump Integration

### Task 2: Add CLI flags and wire dump saving into inference handoff path

**Files:**
- Modify: `scripts/inference_smooth_0912.py`
- Modify: `src/openpi/models_pytorch/rtc_utils.py`

- [ ] **Step 1: Add CLI flags**

Add:
- `--dump_chunk_handoff_enable`
- `--dump_chunk_handoff_dir`
- `--dump_chunk_handoff_step_min`
- `--dump_chunk_handoff_step_max`
- `--dump_chunk_handoff_stride`

Defaults should preserve current behavior when disabled.

- [ ] **Step 2: Identify the single handoff integration point**

Use the existing queue merge / `queue_boundary` location so dumping happens once per real handoff event, not every send tick.

- [ ] **Step 3: Write minimal integration code**

At the handoff point:
- gather `raw_prior`, `raw_new`
- gather `executed_prior`, `executed_new`
- compute metadata:
  - `step_idx`
  - `real_delay`
  - `guided_window_start`
  - `handoff_index`
  - `mode`
- call helper to build payload
- save with `np.savez_compressed(...)`

Filename should be deterministic and sortable, for example:
`step_000123_mode_rtc.npz`

- [ ] **Step 4: Add a lightweight log line**

Print a short message when a dump is saved, including step index and output path.

- [ ] **Step 5: Run syntax verification**

Run:
```bash
docker exec openpi_0207_v3 bash -lc 'cd /workspace/robot_repo/openpi_torch/openpi && PYTHONPATH=/workspace/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python - <<\"PY\"\nimport ast\nfrom pathlib import Path\nfor path in [Path(\"scripts/inference_smooth_0912.py\"), Path(\"src/openpi/models_pytorch/rtc_utils.py\")]:\n    ast.parse(path.read_text(), filename=str(path))\n    print(f\"AST OK: {path}\")\nPY'
```

Expected: `AST OK` for both files.

- [ ] **Step 6: Run focused tests**

Run:
```bash
docker exec openpi_0207_v3 bash -lc 'cd /workspace/robot_repo/openpi_torch/openpi && PYTHONPATH=/workspace/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py'
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi add scripts/inference_smooth_0912.py src/openpi/models_pytorch/rtc_utils.py src/openpi/models_pytorch/rtc_utils_test.py
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi commit -m "feat: dump chunk handoff snapshots"
```

## Chunk 3: Offline Plotting Script

### Task 3: Add plotting script for prior/new handoff visualization

**Files:**
- Create: `scripts/plot_chunk_handoff.py`

- [ ] **Step 1: Write the failing script contract as a manual checklist**

The script must support:
- `--input` for file or directory
- `--output_dir`
- `--space raw|executed|both`
- default 6 joint subplots
- prior/new curves with dashed handoff line

- [ ] **Step 2: Create minimal plotting script**

Implement:
- load one or many `.npz`
- select requested space
- normalize arrays to `(T, D)`
- render 6 arm-joint subplots
- save figures to disk

- [ ] **Step 3: Add figure annotations**

Include:
- step index
- mode
- `real_delay`
- `guided_window_start` when available

- [ ] **Step 4: Run a syntax check**

Run:
```bash
python - <<'PY'
import ast, pathlib
path = pathlib.Path('/home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/scripts/plot_chunk_handoff.py')
ast.parse(path.read_text(), filename=str(path))
print(f'AST OK: {path}')
PY
```

Expected: `AST OK`

- [ ] **Step 5: Smoke-test on one saved `.npz`**

Run:
```bash
python /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi/scripts/plot_chunk_handoff.py --input /path/to/sample_step_xxx.npz --output_dir /tmp/chunk_handoff_plots
```

Expected: one or two PNGs saved successfully.

- [ ] **Step 6: Commit**

```bash
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi add scripts/plot_chunk_handoff.py
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi commit -m "feat: add chunk handoff plotting script"
```

## Chunk 4: End-to-End Validation

### Task 4: Verify runtime dump + offline plot workflow together

**Files:**
- Modify if needed: `scripts/inference_smooth_0912.py`
- Create if helpful: `scripts/debug/...` helper shell script

- [ ] **Step 1: Run one short inference with dumping enabled**

Use an existing debug command and add:
- `--dump_chunk_handoff_enable`
- `--dump_chunk_handoff_dir <debug dir>`

Expected: at least one `.npz` file emitted.

- [ ] **Step 2: Plot the produced snapshots**

Run the plotting script on the output directory.

Expected: saved figures for the selected snapshots.

- [ ] **Step 3: Cross-check with existing diagnostics**

Pick one dumped step and verify the visual handoff mismatch qualitatively agrees with:
- `queue_boundary`
- `overlap_l2_before`
- `overlap_l2_after`

- [ ] **Step 4: Run the full targeted verification suite**

Run:
```bash
docker exec openpi_0207_v3 bash -lc 'cd /workspace/robot_repo/openpi_torch/openpi && PYTHONPATH=/workspace/robot_repo/openpi_torch/openpi/src:$PYTHONPATH python -m pytest -q src/openpi/models_pytorch/rtc_utils_test.py src/openpi/shared/inference_kwargs_test.py'
```

Expected: PASS

- [ ] **Step 5: Commit final integration**

```bash
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi add scripts/inference_smooth_0912.py src/openpi/models_pytorch/rtc_utils.py src/openpi/models_pytorch/rtc_utils_test.py scripts/plot_chunk_handoff.py
git -C /home/hyc/openpi_ws/robot_repo/openpi_torch/openpi commit -m "feat: add chunk handoff visualization workflow"
```
