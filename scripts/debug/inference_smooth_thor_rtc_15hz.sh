#!/usr/bin/env bash
set -euo pipefail

echo "Running THOR RTC profile: 15Hz fallback (action_steps=15, rtc_execution_horizon=20)"

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/0311grad_acc_4_lr_2e-4_1e-6/6709 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 15 \
  --fps 15 \
  --rtc_enable \
  --rtc_execution_horizon 20 \
  --rtc_schedule exp
