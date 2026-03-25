#!/usr/bin/env bash
set -euo pipefail

echo "Running THOR async engineering baseline: 20Hz, action_steps=20, cubic transition + EMA horizon smoothing, no RTC"

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/formal_8gpu_bs512_from_30000/9000 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 20 \
  --smooth_type cubic \
  --horizon_smooth ema \
  --horizon_ema_alpha 0.5 \
  --qp_lambda_acc 0 \
  --dump_chunk_handoff_enable \
  --dump_chunk_handoff_dir scripts/debug/chunk_handoff_dumps_async_postprocess_20hz_20step_9000
