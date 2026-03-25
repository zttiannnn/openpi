#!/usr/bin/env bash
set -euo pipefail

echo "Running raw_prefix_paper RTC: w=10 sigma_d=0.5"

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/formal_8gpu_bs512_from_30000/30000 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 5 \
  --smooth_type none \
  --horizon_smooth none \
  --rtc_enable \
  --rtc_guidance_mode raw_prefix_paper \
  --rtc_schedule auto \
  --rtc_max_guidance_weight 10.0 \
  --rtc_sigma_d 0.5 \
  --execution_interp_enable \
  --execution_send_fps 20 \
  --execution_interp_method linear \
  --qp_lambda_acc 0 \
  --dump_chunk_handoff_enable \
  --dump_chunk_handoff_dir scripts/debug/chunk_handoff_dumps_async_raw_paper_rtc_w10_sd050_30000_5interp20
