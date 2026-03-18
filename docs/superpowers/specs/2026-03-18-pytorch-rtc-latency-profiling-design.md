# PyTorch RTC Latency Profiling Design

**Date:** 2026-03-18

**Goal**

Add a profiling path for PyTorch `pi0.5` inference that can measure and compare latency breakdowns for vanilla inference and RTC/B1 inference on the same codepath and with the same timing semantics.

**Why**

Current logs only expose coarse end-to-end inference time. That is enough to know RTC/B1 is slower, but not enough to answer:

- whether the slowdown is dominated by loss of `torch.compile`
- whether RTC/B1 overhead mainly comes from executed-space decode, loss evaluation, autograd, or diagnostics
- whether Jetson Thor timings are directionally consistent with the paper's latency table

This profiling feature is intended to answer those questions on Jetson first and later be reusable on RTX 4090 without changing the instrumentation model.

## Scope

In scope:

- PyTorch inference only
- `sample_actions()` and `sample_actions_rtc()`
- single-trace output and multi-run aggregate output
- terminal summary plus structured JSON/CSV outputs
- component timings for both no-RTC and RTC

Out of scope:

- JAX profiling
- camera/ROS/control-loop timing
- replacing the current RTC algorithm
- using `torch.profiler` as the primary solution

## Profiling Model

The design uses a lightweight manual profiler with explicit start/stop events around semantically meaningful regions. Timings are collected only when profiling is enabled and are synchronized on CUDA boundaries so GPU timings are comparable across components.

The profiler records both:

- a single full trace for one run
- aggregate summaries over N measured runs after warmup

## Breakdown Semantics

To stay close to the paper while still being useful for debugging, the profiler reports:

- `image_encoders_ms`
  - image embedding work inside `embed_prefix()`
- `llm_prefill_ms`
  - prefix cache prefill forward pass
- `denoising_total_ms`
  - sum over all denoising iterations
- `denoising_step_mean_ms`
  - average denoising step cost
- `denoising_step_trace_ms`
  - per-step list
- `total_model_ms`
  - end-to-end model sampling cost

RTC-only additions:

- `rtc_guidance_total_ms`
- `rtc_decode_ms`
- `rtc_loss_ms`
- `rtc_autograd_ms`
- `rtc_diag_ms`

These preserve a paper-like top-level table while exposing the RTC-internal breakdown that the paper does not show.

## Instrumentation Points

Primary code locations:

- `src/openpi/models_pytorch/pi0_pytorch.py`
  - `embed_prefix()`
  - `sample_actions()`
  - `sample_actions_rtc()`
  - `denoise_step()`
- `src/openpi/models_pytorch/rtc_utils.py`
  - inside `apply_rtc_guidance()` around decode/loss/autograd/diagnostics segments
- `src/openpi/policies/policy.py`
  - attach profiling payload to policy outputs
- `scripts/inference_smooth_0912.py`
  - CLI flags and final printing/file output

## Output Contract

When profiling is enabled, policy outputs include a structured payload with:

- mode: `no_rtc` or `rtc`
- component totals
- optional per-step traces
- trace metadata:
  - `num_steps`
  - `warmup_runs`
  - `measured_runs`
  - device info

CLI output modes:

- single trace human-readable table
- aggregate mean/std table
- JSON trace file
- CSV summary rows

## Consistency Rules

To keep no-RTC and RTC comparable:

- both paths use the same profiler object and field names
- `total_model_ms` is measured over the same sample function boundary
- CUDA synchronization is applied uniformly when profiling is enabled
- diagnostics cost is isolated into `rtc_diag_ms` rather than folded into decode/loss/autograd

## Risk Management

Potential risks:

- profiling overhead perturbing absolute timings
- compile interaction on the no-RTC path
- accidental measurement drift between no-RTC and RTC

Mitigations:

- profiling is opt-in only
- warmup runs are configurable
- single-trace and aggregate outputs are both recorded
- tests verify field presence and aggregation semantics

## Success Criteria

The feature is successful if we can run one command for no-RTC and one for RTC and get:

1. a top-level table comparable to the paper's latency breakdown
2. RTC-internal timing breakdown for decode/loss/autograd/diagnostics
3. structured JSON/CSV outputs for later comparison on Jetson and 4090
