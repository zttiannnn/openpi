# B1 Strength Sweep Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small three-way B1 strength sweep around the current `s08_p3` latency baseline so robot tests can compare weaker, baseline, and stronger guidance without changing other RTC timing knobs.

**Architecture:** Keep the existing `s08_p3` script as the middle setting. Add one weaker and one stronger shell script that differ only in `rtc_max_guidance_weight` and `rtc_b1_guided_delta_abs_max`. Do not touch model logic or runtime plumbing.

**Tech Stack:** Bash, existing `scripts/inference_smooth_0912.py` CLI, shell syntax validation with `bash -n`

---

## Chunk 1: Script Sweep

### Task 1: Document the sweep shape

**Files:**
- Create: `openpi/docs/superpowers/plans/2026-03-17-b1-strength-sweep.md`

- [ ] **Step 1: Record the sweep design**

Write down the three settings:
- weak: lower `rtc_max_guidance_weight` and lower `rtc_b1_guided_delta_abs_max`
- base: keep `inference_smooth_thor_rtc1_latency_s08_p3.sh` unchanged
- strong: higher `rtc_max_guidance_weight` and higher `rtc_b1_guided_delta_abs_max`

- [ ] **Step 2: Keep all other knobs fixed**

Do not change:
- `num_steps`
- `rtc_guidance_start_fraction`
- `rtc_prefix_steps`
- `horizon_smooth`
- `action_steps`
- `fps`

### Task 2: Add the weak and strong scripts

**Files:**
- Modify: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3.sh`
- Create: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_weak.sh`
- Create: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_strong.sh`

- [ ] **Step 1: Keep the current base script unchanged**

Use the existing middle setting:

```bash
--rtc_max_guidance_weight 4.0
--rtc_b1_guided_delta_abs_max 20.0
```

- [ ] **Step 2: Add the weak script**

Use:

```bash
--rtc_max_guidance_weight 2.0
--rtc_b1_guided_delta_abs_max 10.0
```

- [ ] **Step 3: Add the strong script**

Use:

```bash
--rtc_max_guidance_weight 6.0
--rtc_b1_guided_delta_abs_max 30.0
```

- [ ] **Step 4: Keep everything else identical to the base script**

This makes the logs directly comparable.

### Task 3: Validate the scripts

**Files:**
- Test: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3.sh`
- Test: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_weak.sh`
- Test: `openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_strong.sh`

- [ ] **Step 1: Run shell syntax checks**

Run:

```bash
bash -n openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3.sh
bash -n openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_weak.sh
bash -n openpi/scripts/inference_smooth_thor_rtc1_latency_s08_p3_strong.sh
```

Expected: no output, exit code `0`

- [ ] **Step 2: Hand off the run order**

Recommend running:

```bash
bash scripts/inference_smooth_thor_rtc1_latency_s08_p3_weak.sh
bash scripts/inference_smooth_thor_rtc1_latency_s08_p3.sh
bash scripts/inference_smooth_thor_rtc1_latency_s08_p3_strong.sh
```

Expected comparison targets:
- `infer time`
- `real_delay`
- `RTC B1 boundary window`
- `queue_boundary`
