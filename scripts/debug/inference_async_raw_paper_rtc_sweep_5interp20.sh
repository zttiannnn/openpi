#!/usr/bin/env bash
set -euo pipefail

# Sweep the two raw RTC strength knobs that matter most now that paper-style
# overlap guidance is wired in:
#   1. rtc_max_guidance_weight
#   2. rtc_sigma_d
#
# The script runs each configuration sequentially and saves a dedicated log file
# plus handoff dump directory for later comparison.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

weights=(10 20 30)
sigmas=(1.0 0.5 0.25)

format_sigma_tag() {
  case "$1" in
    1.0) echo "100" ;;
    0.5) echo "050" ;;
    0.25) echo "025" ;;
    *) echo "${1//./}" ;;
  esac
}

run_one() {
  local weight="$1"
  local sigma="$2"
  local sigma_tag
  sigma_tag="$(format_sigma_tag "$sigma")"

  local log_path="scripts/debug/log_async_raw_paper_rtc_w${weight}_sd${sigma_tag}_30000_5interp20.txt"
  local dump_dir="scripts/debug/chunk_handoff_dumps_async_raw_paper_rtc_w${weight}_sd${sigma_tag}_30000_5interp20"

  mkdir -p "$dump_dir"

  echo
  echo "=== raw_prefix_paper sweep: max_guidance_weight=${weight}, sigma_d=${sigma} ==="
  echo "log:  ${log_path}"
  echo "dump: ${dump_dir}"

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
    --rtc_max_guidance_weight "$weight" \
    --rtc_sigma_d "$sigma" \
    --execution_interp_enable \
    --execution_send_fps 20 \
    --execution_interp_method linear \
    --qp_lambda_acc 0 \
    --dump_chunk_handoff_enable \
    --dump_chunk_handoff_dir "$dump_dir" \
    2>&1 | tee "$log_path"
}

for weight in "${weights[@]}"; do
  for sigma in "${sigmas[@]}"; do
    run_one "$weight" "$sigma"
  done
done

echo
echo "Sweep finished. Logs are under scripts/debug/log_async_raw_paper_rtc_w*_sd*_30000_5interp20.txt"
