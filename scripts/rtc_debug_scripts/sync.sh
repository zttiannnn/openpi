#!/usr/bin/env bash
set -euo pipefail

echo "Running sync inference"

python -m scripts.inference_smooth_0912 \
  --port can_left \
  --id left \
  --config pi05_agileX_thor_jax \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor_jax/from_1206_pi05_jax/25000 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 50 \
  --fps 20 \
  --smooth_type none \
  --horizon_smooth ema \
  --qp_lambda_acc 0 \
  --dump_chunk_handoff_enable \
  --dump_chunk_handoff_dir scripts/debug/sync
