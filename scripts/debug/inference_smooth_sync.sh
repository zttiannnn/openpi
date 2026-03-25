#!/usr/bin/env bash
set -euo pipefail

echo "Running THOR sync"

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/0311grad_acc_4_lr_2e-4_1e-6/11106 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 50 \
  --fps 20 \
  --smooth_type none \
  --horizon_smooth ema \
  --horizon_ema_alpha 0.5 \
  --print_action_arrays \
  --print_action_array_rows 3 \
  --qp_lambda_acc 0 \
  --profile_model_enable \
  --profile_model_warmup_runs 1 \
  --profile_model_runs 5 \
  --profile_model_output_json openpi/scripts/debug/profile_no_rtc.json \
  --profile_model_output_csv openpi/scripts/debug/profile_no_rtc.csv
