#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

echo "Running Async inference"

DELAY_HISTOGRAM_OUTPUT_JSON="${DELAY_HISTOGRAM_OUTPUT_JSON:-$ROOT_DIR/scripts/debug/Async/delay_histogram_async_fps30.json}"

python -m scripts.inference_smooth_0912 \
  --port can_left \
  --id left \
  --config pi05_agileX_thor_jax \
  --checkpoint_dir ./checkpoints/pi05_agileX_thor_jax/from_1206_pi05_jax/59999 \
  --task "Put the purple carton of milk into the cardboard box." \
  --cameras '{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}' \
  --action_steps 20 \
  --fps 30 \
  --smooth_type cubic \
  --horizon_smooth ema \
  --qp_lambda_acc 0 \
  --delay_histogram_output_json "$DELAY_HISTOGRAM_OUTPUT_JSON" \
  --dump_chunk_handoff_enable \
  --dump_chunk_handoff_dir scripts/debug/Async
