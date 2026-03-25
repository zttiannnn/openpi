#!/usr/bin/env bash
set -euo pipefail

python scripts/inference_smooth_0912.py \
  --port can_left \
  --id left \
  --config pi05_agileX_thor \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor/0311grad_acc_4_lr_2e-4_1e-6/11106 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 20 \
  --smooth_type none \
  --horizon_smooth ema \
  --horizon_ema_alpha 0.5 \
  --rtc_enable \
  --rtc_guidance_mode executed_prefix_b1 \
  --rtc_guidance_start_fraction 0.8 \
  --rtc_prefix_steps 3 \
  --rtc_max_guidance_weight 6.0 \
  --rtc_stay_weight 0.0 \
  --rtc_b1_guided_delta_abs_max 30.0 \
  --qp_lambda_acc 0
