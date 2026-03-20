# Resume 8-GPU Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable debug script that resumes the current 8-GPU PyTorch training experiment with the required environment setup.

**Architecture:** The implementation is a single shell script in `scripts/debug/` plus brief design and plan documents in `docs/superpowers/`. The script centralizes the validated `torchrun` command and exposes the most likely tuning parameters as top-level variables.

**Tech Stack:** Bash, `torchrun`, existing `openpi` PyTorch training entrypoint.

---

### Task 1: Add reusable resume script

**Files:**
- Create: `scripts/debug/resume_8gpu.sh`
- Create: `docs/superpowers/specs/2026-03-20-resume-8gpu-training-design.md`
- Create: `docs/superpowers/plans/2026-03-20-resume-8gpu-training.md`

- [ ] **Step 1: Document the intended command structure**

Write a short design note describing why the script should use `--resume`, preserve the existing `exp_name`, and carry the required `PYTHONPATH`/`NCCL_NVLS_ENABLE=0` environment.

- [ ] **Step 2: Add the shell script**

Create a focused Bash script that:
- activates the shared virtualenv,
- enters the project root,
- exports the required `PYTHONPATH`,
- launches `torchrun` with the existing 8-GPU resume configuration.

- [ ] **Step 3: Sanity-check the script contents**

Verify the script references the intended experiment name, dataset paths, and resume flag, and that its defaults match the agreed training setup.
