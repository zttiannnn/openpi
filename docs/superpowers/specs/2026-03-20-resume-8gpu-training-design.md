# Resume 8-GPU Training Script Design

**Goal:** Add a reusable shell script that resumes the current 8-GPU PyTorch training run from the latest checkpoint in the existing experiment directory.

**Context:** The current PyTorch training setup requires several environment overrides (`PYTHONPATH`, `NCCL_NVLS_ENABLE=0`) and a long `torchrun` command. The experiment already produced checkpoints under a fixed `exp_name`, so continuing should use `--resume` instead of reloading `--pytorch_weight_path`.

**Approach:**
- Add a dedicated debug shell script under `scripts/debug/` that mirrors the manually validated command line.
- Expose the most useful knobs as variables at the top of the script: experiment name, dataset paths, batch size, num workers, step count, save interval, and GPU list.
- Default `num_workers` to `4` as a middle ground between the previously stable-but-slower `2` and the higher-throughput-but-riskier `8`.
- Keep log output streamed to stdout so the caller can choose `tee`, `nohup`, or other wrappers externally.

**Non-Goals:**
- No code changes to the training loop itself.
- No automated retry or watchdog wrapper.
- No changes to the shared server environment.
