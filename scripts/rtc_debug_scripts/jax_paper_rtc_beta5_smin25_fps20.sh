#!/usr/bin/env bash
set -euo pipefail

# JAX paper RTC launcher for pi05_agileX_thor_jax.
# Override values at runtime, for example:
#   CHECKPOINT_DIR=/abs/path/to/60000 TASK='your task' bash scripts/rtc_debug_scripts/jax_paper_rtc_beta5_smin25_fps30.sh
#   CAMERAS_FILE=/abs/path/to/cameras.yaml PORT=can_right ARM_ID=right bash scripts/rtc_debug_scripts/jax_paper_rtc_beta5_smin25_fps30.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
CONFIG="${CONFIG:-pi05_agileX_thor_jax}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ROOT_DIR/checkpoints/pi05_agileX_thor_jax/from_1206_pi05_jax/59999}"
PORT="${PORT:-can_left}"
ARM_ID="${ARM_ID:-left}"
TASK="${TASK:-Put the purple carton of milk into the cardboard box.}"
FPS="${FPS:-20}"
NUM_STEPS="${NUM_STEPS:-5}"
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
DUMP_CHUNK_HANDOFF_DIR="${DUMP_CHUNK_HANDOFF_DIR:-$ROOT_DIR/scripts/debug/jax_chunk_handoff_dumps_fps20}"
PLOT_CHUNK_HANDOFF_ENABLE="${PLOT_CHUNK_HANDOFF_ENABLE:-0}"
PLOT_CHUNK_HANDOFF_DIR="${PLOT_CHUNK_HANDOFF_DIR:-$ROOT_DIR/scripts/debug/jax_chunk_handoff_plots_fps20}"

if [[ -n "$CAMERAS_FILE" ]]; then
  CAMERAS="$(cat "$CAMERAS_FILE")"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERR] Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -d "$CHECKPOINT_DIR/params" ]]; then
  echo "[ERR] JAX checkpoint params directory not found: $CHECKPOINT_DIR/params" >&2
  exit 1
fi

if [[ -f "$CHECKPOINT_DIR/model.safetensors" ]]; then
  echo "[ERR] Found model.safetensors in $CHECKPOINT_DIR; this launcher is for JAX checkpoints only." >&2
  exit 1
fi

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
if [[ "$DUMP_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
  EXTRA_ARGS+=(--dump_chunk_handoff_enable --dump_chunk_handoff_dir "$DUMP_CHUNK_HANDOFF_DIR")
fi
if [[ "$PLOT_CHUNK_HANDOFF_ENABLE" == "1" ]]; then
  EXTRA_ARGS+=(--plot_chunk_handoff_enable --plot_chunk_handoff_dir "$PLOT_CHUNK_HANDOFF_DIR")
fi

export XLA_PYTHON_CLIENT_PREALLOCATE
export XLA_PYTHON_CLIENT_MEM_FRACTION
export XLA_FLAGS

echo "Running JAX paper RTC launcher"
echo "  config=$CONFIG"
echo "  checkpoint_dir=$CHECKPOINT_DIR"
echo "  port=$PORT id=$ARM_ID"
echo "  fps=$FPS num_steps=$NUM_STEPS"
echo "  rtc_beta=$RTC_BETA rtc_s_min=$RTC_S_MIN rtc_initial_delay=$RTC_INITIAL_DELAY rtc_delay_buffer_size=$RTC_DELAY_BUFFER_SIZE"
echo "  XLA_FLAGS=$XLA_FLAGS"
echo "  dump_chunk_handoff_enable=$DUMP_CHUNK_HANDOFF_ENABLE dump_dir=$DUMP_CHUNK_HANDOFF_DIR"
echo "  plot_chunk_handoff_enable=$PLOT_CHUNK_HANDOFF_ENABLE plot_dir=$PLOT_CHUNK_HANDOFF_DIR"

"$PYTHON_BIN" -m scripts.inference_jax_rtc_0325 \
  --port "$PORT" \
  --id "$ARM_ID" \
  --config "$CONFIG" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --task "$TASK" \
  --cameras "$CAMERAS" \
  --fps "$FPS" \
  --num_steps "$NUM_STEPS" \
  --rtc_beta "$RTC_BETA" \
  --rtc_s_min "$RTC_S_MIN" \
  --rtc_initial_delay "$RTC_INITIAL_DELAY" \
  --rtc_delay_buffer_size "$RTC_DELAY_BUFFER_SIZE" \
  --seed "$SEED" \
  --max_steps "$MAX_STEPS" \
  "${EXTRA_ARGS[@]}"
