#!/usr/bin/env bash
set -euo pipefail

echo "Running THOR async raw-prefix RTC comparison: 20Hz, action_steps=20, raw_prefix RTC, no horizon smoothing"

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/formal_8gpu_bs512_from_30000/9000 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 5 \
  --smooth_type none \
  --horizon_smooth none \
  --rtc_enable \
  --rtc_guidance_mode raw_prefix \
  --rtc_execution_horizon 8 \
  --rtc_max_guidance_weight 4.0 \
  --execution_interp_enable \
  --execution_send_fps 20 \
  --execution_interp_method linear \
  --qp_lambda_acc 0 \
  --dump_chunk_handoff_enable \
  --dump_chunk_handoff_dir scripts/debug/chunk_handoff_dumps_async_raw_rtc_20hz_20step_9000_5interp20
