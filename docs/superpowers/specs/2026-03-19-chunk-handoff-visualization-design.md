# Chunk Handoff Visualization Design

## Goal

Provide an offline visualization workflow that makes chunk-to-chunk disagreement visible at the handoff boundary without adding meaningful overhead to model inference or robot control.

The immediate goal is to answer questions like:

- How different are the prior and new chunks near the handoff step?
- Is the disagreement already present in raw model output, or mostly introduced later in the execution pipeline?
- How large is the mismatch in the final executed/action-queue space compared with the raw chunk space?

## Scope

This design covers:

- Runtime dumping of compact handoff snapshots to `.npz`
- An offline plotting script that renders prior/new action curves around the handoff
- Support for both raw model-space chunks and executed/action-queue chunks

This design does **not** cover:

- End-effector `xyz` plotting in v1
- Real-time plotting during inference
- Any new smoothing or control behavior

## Approach

### Recommended approach

Use a two-stage workflow:

1. During inference, save lightweight handoff snapshots whenever a new chunk is merged or takes over.
2. After the run, use a standalone plotting script to render the saved snapshots as multi-panel prior/new joint curves.

This keeps runtime overhead low and avoids contaminating latency experiments while still giving us rich post-hoc inspection.

### Alternatives considered

- Parse existing logs:
  - Rejected because current logs only contain previews/head-tail snippets and are too brittle for full-curve plots.
- Plot at runtime:
  - Rejected because `matplotlib` and image saving would interfere with latency and control behavior.

## Data Model

Each saved `.npz` snapshot should contain one handoff event.

### Required metadata

- `step_idx`: inference step index
- `mode`: `rtc` or `no_rtc`
- `real_delay`
- `guided_window_start` if available, else `-1`
- `handoff_index`: the step index within the plotted window where the new chunk takes over

### Required arrays

- `raw_prior`: prior/raw leftover chunk relevant to the handoff
- `raw_new`: new/raw chunk from the current inference
- `executed_prior`: prior chunk in the executed/action-queue space
- `executed_new`: new chunk in the executed/action-queue space

### Shape conventions

All chunk arrays should be stored as `(T, D)` float arrays.

- `T`: number of steps in the stored chunk segment
- `D`: action dimension

The plotting script will focus on the first 6 arm-joint dimensions by default.

## Runtime Dumping

Dumping should happen only at chunk handoff points, not every control step.

### CLI controls

Add optional debug-only flags to `scripts/inference_smooth_0912.py`:

- `--dump_chunk_handoff_enable`
- `--dump_chunk_handoff_dir`
- `--dump_chunk_handoff_step_min`
- `--dump_chunk_handoff_step_max`
- `--dump_chunk_handoff_stride`

### Behavior

- If dumping is disabled, runtime behavior remains unchanged.
- If enabled, the script saves one `.npz` per qualifying handoff event.
- Filtering rules:
  - step range filter
  - stride filter
- Dumping should reuse arrays already available in the handoff/queue merge logic instead of recomputing extra transforms.

## Plotting Script

Create a standalone script dedicated to handoff visualization.

### Input

- One `.npz` file, or a directory of `.npz` files

### Output

By default, produce:

- one figure for `raw_*`
- one figure for `executed_*`

Each figure contains 6 subplots, one per arm joint dimension.

### Plot style

For each subplot:

- blue line: `Prior`
- red line: `New`
- gray dashed vertical line: handoff index

Optional legend/title information:

- `step_idx`
- `real_delay`
- `guided_window_start`
- per-dimension boundary delta at handoff

### Comparison semantics

The script should plot the prior and new curves on a shared step axis so the handoff discontinuity is visually obvious.

The handoff index should align with the effective boundary:

- for RTC/no-RTC raw chunk comparison: the chunk boundary relevant to the current merge
- for executed chunk comparison: the first step where the new queue head takes over

## Testing Strategy

### Unit tests

Add focused tests for:

- handoff dump payload construction
- step-range and stride filtering
- shape normalization to `(T, D)`

### Manual validation

Use one RTC run and one no-RTC run to verify:

- `.npz` files are produced only for qualifying handoff events
- `raw` figures and `executed` figures both render
- the plotted handoff discontinuity matches the existing `queue_boundary` / overlap diagnostics qualitatively

## Expected Outcome

After this lands, we should be able to inspect chunk consistency visually instead of relying only on scalar logs.

That should let us separate three cases much more clearly:

- raw chunk disagreement already large
- raw disagreement modest but executed-space disagreement large
- disagreement localized near handoff vs. disagreement spread across the whole new chunk

This visualization is intended as a diagnostic tool for deciding whether further effort should stay in inference-time RTC/B1 or move to model/training-side chunk consistency work.
