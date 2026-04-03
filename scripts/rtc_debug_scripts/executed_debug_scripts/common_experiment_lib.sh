#!/usr/bin/env bash
set -euo pipefail

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$COMMON_DIR/../../.." && pwd)"

rtc_experiment_init() {
  local script_path="$1"
  SCRIPT_NAME="$(basename "$script_path" .sh)"

  cd "$ROOT_DIR"

  PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
  CONFIG="${CONFIG:-pi05_agileX_thor_jax}"
  CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/pi05_agileX_thor_jax/from_1206_pi05_jax/59999}"
  PORT="${PORT:-can_left}"
  ARM_ID="${ARM_ID:-left}"
  TASK="${TASK:-Put the purple carton of milk into the cardboard box.}"
  FPS="${FPS:-20}"
  NUM_STEPS="${NUM_STEPS:-5}"
  ACTION_STEPS="${ACTION_STEPS:-20}"
  RTC_BETA="${RTC_BETA:-5.0}"
  RTC_S_MIN="${RTC_S_MIN:-25}"
  RTC_INITIAL_DELAY="${RTC_INITIAL_DELAY:-0}"
  RTC_DELAY_BUFFER_SIZE="${RTC_DELAY_BUFFER_SIZE:-10}"
  SEED="${SEED:-10002}"
  MAX_STEPS="${MAX_STEPS:-600000}"
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
  XLA_FLAGS="${XLA_FLAGS:---xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0}"

  CAMERAS_DEFAULT='{camera0: {type: orbbec, index_or_path: CP0H953000YB, width: 640, height: 480, fps: 30}, camera1: {type: orbbec, index_or_path: CP02653000TL, width: 640, height: 480, fps: 30}}'
  CAMERAS="${CAMERAS:-$CAMERAS_DEFAULT}"
  CAMERAS_FILE="${CAMERAS_FILE:-}"
  USE_DEGREES="${USE_DEGREES:-0}"
  MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"

  PRINT_ACTION_ARRAYS="${PRINT_ACTION_ARRAYS:-0}"
  DUMP_CHUNK_HANDOFF_ENABLE="${DUMP_CHUNK_HANDOFF_ENABLE:-1}"
  PLOT_CHUNK_HANDOFF_ENABLE="${PLOT_CHUNK_HANDOFF_ENABLE:-1}"

  EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ROOT_DIR/scripts/debug/executed_debug_scripts}"
  EXPERIMENT_DIR="${EXPERIMENT_DIR:-$EXPERIMENT_ROOT/$SCRIPT_NAME}"
  DUMP_CHUNK_HANDOFF_DIR="${DUMP_CHUNK_HANDOFF_DIR:-$EXPERIMENT_DIR/dumps}"
  PLOT_CHUNK_HANDOFF_DIR="${PLOT_CHUNK_HANDOFF_DIR:-$EXPERIMENT_DIR/plots}"

  if [[ -n "$CAMERAS_FILE" ]]; then
    CAMERAS="$(cat "$CAMERAS_FILE")"
  fi
}

rtc_require_python_and_checkpoint() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERR] Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
  fi

  if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
    echo "[ERR] JAX checkpoint params directory not found: $CHECKPOINT_DIR/params" >&2
    exit 1
  fi

  if [[ -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
    echo "[ERR] Found model.safetensors in $CHECKPOINT_DIR; these launchers are for JAX checkpoints only." >&2
    exit 1
  fi
}

rtc_prepare_output_dirs() {
  if [[ "$DUMP_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
    mkdir -p "$DUMP_CHUNK_HANDOFF_DIR"
  fi
  if [[ "$PLOT_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
    mkdir -p "$PLOT_CHUNK_HANDOFF_DIR"
  fi
}

rtc_common_extra_args() {
  EXTRA_ARGS=()
  if [[ "$USE_DEGREES" == "1" ]]; then
    EXTRA_ARGS+=(--use_degrees)
  fi
  if [[ -n "$MAX_RELATIVE_TARGET" ]]; then
    EXTRA_ARGS+=(--max_relative_target "$MAX_RELATIVE_TARGET")
  fi
  if [[ "$PRINT_ACTION_ARRAYS" == "1" ]]; then
    EXTRA_ARGS+=(--print_action_arrays)
  fi
}

rtc_dump_extra_args() {
  DUMP_ARGS=()
  if [[ "$DUMP_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
    DUMP_ARGS+=(--dump_chunk_handoff_enable --dump_chunk_handoff_dir "$DUMP_CHUNK_HANDOFF_DIR")
  fi
  if [[ "$PLOT_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
    DUMP_ARGS+=(--plot_chunk_handoff_enable --plot_chunk_handoff_dir "$PLOT_CHUNK_HANDOFF_DIR")
  fi
}

rtc_print_banner() {
  local label="$1"
  echo "Running $label"
  echo "  script=$SCRIPT_NAME"
  echo "  config=$CONFIG"
  echo "  checkpoint_dir=$CHECKPOINT_DIR"
  echo "  port=$PORT id=$ARM_ID"
  echo "  fps=$FPS"
  echo "  dump_chunk_handoff_enable=$DUMP_CHUNK_HANDOFF_ENABLE dump_dir=$DUMP_CHUNK_HANDOFF_DIR"
}

run_sync_experiment() {
  rtc_require_python_and_checkpoint
  rtc_prepare_output_dirs
  rtc_common_extra_args
  rtc_dump_extra_args
  rtc_print_banner "sync inference reference"
  echo "  action_steps=$ACTION_STEPS smooth_type=none horizon_smooth=ema"

  "$PYTHON_BIN" -m scripts.inference_smooth_0912 \
    --port "$PORT" \
    --id "$ARM_ID" \
    --config "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task "$TASK" \
    --cameras "$CAMERAS" \
    --action_steps "$ACTION_STEPS" \
    --fps "$FPS" \
    --smooth_type none \
    --horizon_smooth ema \
    --qp_lambda_acc 0 \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    "${DUMP_ARGS[@]}"
}

run_async_experiment() {
  rtc_require_python_and_checkpoint
  rtc_prepare_output_dirs
  rtc_common_extra_args
  rtc_dump_extra_args
  rtc_print_banner "async inference reference"
  echo "  action_steps=$ACTION_STEPS smooth_type=cubic horizon_smooth=ema"

  "$PYTHON_BIN" -m scripts.inference_smooth_0912 \
    --port "$PORT" \
    --id "$ARM_ID" \
    --config "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task "$TASK" \
    --cameras "$CAMERAS" \
    --action_steps "$ACTION_STEPS" \
    --fps "$FPS" \
    --smooth_type cubic \
    --horizon_smooth ema \
    --qp_lambda_acc 0 \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    "${DUMP_ARGS[@]}"
}

run_jax_rtc_experiment() {
  local rtc_mode="$1"

  rtc_require_python_and_checkpoint
  rtc_prepare_output_dirs
  rtc_common_extra_args
  rtc_dump_extra_args

  export XLA_PYTHON_CLIENT_PREALLOCATE
  export XLA_PYTHON_CLIENT_MEM_FRACTION
  export XLA_FLAGS

  rtc_print_banner "JAX RTC experiment"
  echo "  rtc_mode=$rtc_mode num_steps=$NUM_STEPS rtc_beta=$RTC_BETA rtc_s_min=$RTC_S_MIN"
  echo "  rtc_initial_delay=$RTC_INITIAL_DELAY rtc_delay_buffer_size=$RTC_DELAY_BUFFER_SIZE"
  echo "  plot_chunk_handoff_enable=$PLOT_CHUNK_HANDOFF_ENABLE plot_dir=$PLOT_CHUNK_HANDOFF_DIR"
  echo "  XLA_FLAGS=$XLA_FLAGS"

  "$PYTHON_BIN" -m scripts.inference_jax_rtc_0325 \
    --port "$PORT" \
    --id "$ARM_ID" \
    --config "$CONFIG" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --task "$TASK" \
    --cameras "$CAMERAS" \
    --fps "$FPS" \
    --num_steps "$NUM_STEPS" \
    --rtc_mode "$rtc_mode" \
    --rtc_beta "$RTC_BETA" \
    --rtc_s_min "$RTC_S_MIN" \
    --rtc_initial_delay "$RTC_INITIAL_DELAY" \
    --rtc_delay_buffer_size "$RTC_DELAY_BUFFER_SIZE" \
    --seed "$SEED" \
    --max_steps "$MAX_STEPS" \
    "${EXTRA_ARGS[@]}" \
    "${DUMP_ARGS[@]}"
}
