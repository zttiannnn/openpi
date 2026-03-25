#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DEFAULT_PROJECT_VENV="${PROJECT_DIR}/.venv"
DEFAULT_SHARED_VENV="/jedata/jemotor/code/openpi/.venv"

if [[ -n "${VENV_PATH:-}" ]]; then
  VENV_PATH="${VENV_PATH}"
elif [[ -f "${DEFAULT_PROJECT_VENV}/bin/activate" ]]; then
  VENV_PATH="${DEFAULT_PROJECT_VENV}"
elif [[ -f "${DEFAULT_SHARED_VENV}/bin/activate" ]]; then
  VENV_PATH="${DEFAULT_SHARED_VENV}"
else
  VENV_PATH="${DEFAULT_PROJECT_VENV}"
fi

CONFIG_NAME="${CONFIG_NAME:-pi05_agileX_thor_jax}"
EXP_NAME="${EXP_NAME:-from_1206_pi05_jax}"
DATA_ROOT="${DATA_ROOT:-/jedata/jemotor/code/openpi_pi05_torch_1/JE_robot_data_lerobot/0311_data_lerobot}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-Put the purple carton of milk into the cardboard box.}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"

BATCH_SIZE="${BATCH_SIZE:-512}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-60000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
PEAK_LR="${PEAK_LR:-2e-4}"
DECAY_LR="${DECAY_LR:-1e-6}"

CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-${PROJECT_DIR}/checkpoints}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-checkpoints/pi05_agileX_thor/1206_pi05_jax}"
BASE_CKPT_STEP="${BASE_CKPT_STEP:-latest}"
FSDP_DEVICES="${FSDP_DEVICES:-}"

RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
WANDB_ENABLED="${WANDB_ENABLED:-0}"

if [[ "${OVERWRITE}" == "1" && "${RESUME}" == "1" ]]; then
  echo "ERROR: RESUME and OVERWRITE cannot both be 1." >&2
  exit 1
fi

count_visible_devices() {
  local devices="$1"
  local -a parsed_devices=()
  IFS=',' read -r -a parsed_devices <<< "${devices}"
  printf '%s\n' "${#parsed_devices[@]}"
}

VISIBLE_DEVICE_COUNT="$(count_visible_devices "${CUDA_DEVICES}")"
if [[ -z "${FSDP_DEVICES}" ]]; then
  FSDP_DEVICES="${VISIBLE_DEVICE_COUNT}"
fi

if [[ "${FSDP_DEVICES}" != "${VISIBLE_DEVICE_COUNT}" ]]; then
  cat >&2 <<ERRMSG
ERROR: FSDP_DEVICES=${FSDP_DEVICES} does not match the number of CUDA devices in CUDA_DEVICES=${CUDA_DEVICES} (${VISIBLE_DEVICE_COUNT}).

Please keep these in sync. For example:
  CUDA_DEVICES=0,1,2,3,4,5,6,7 FSDP_DEVICES=8 bash scripts/train_scripts/train_jax.sh
ERRMSG
  exit 1
fi

resolve_weight_loader_path() {
  local base_path="$1"
  local ckpt_step="$2"

  if [[ "${base_path}" == gs://* ]]; then
    if [[ "${base_path}" == */params ]]; then
      printf '%s\n' "${base_path}"
    else
      printf '%s/params\n' "${base_path}"
    fi
    return 0
  fi

  if [[ "${base_path}" == */params ]]; then
    printf '%s\n' "${base_path}"
    return 0
  fi

  if [[ -d "${base_path}/params" ]]; then
    printf '%s\n' "${base_path}/params"
    return 0
  fi

  if [[ "${ckpt_step}" != "latest" && -d "${base_path}/${ckpt_step}/params" ]]; then
    printf '%s\n' "${base_path}/${ckpt_step}/params"
    return 0
  fi

  local latest_step=""
  if [[ -d "${base_path}" ]]; then
    while IFS= read -r step_dir; do
      latest_step="$(basename "${step_dir}")"
    done < <(find "${base_path}" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' | sort -V)
  fi

  if [[ -n "${latest_step}" && -d "${base_path}/${latest_step}/params" ]]; then
    printf '%s\n' "${base_path}/${latest_step}/params"
    return 0
  fi

  return 1
}

if [[ "${RESUME}" != "1" ]]; then
  if [[ "${BASE_MODEL_PATH}" != /* && "${BASE_MODEL_PATH}" != gs://* ]]; then
    BASE_MODEL_PATH="${PROJECT_DIR}/${BASE_MODEL_PATH}"
  fi

  if ! WEIGHT_LOADER_PATH="$(resolve_weight_loader_path "${BASE_MODEL_PATH}" "${BASE_CKPT_STEP}")"; then
    cat >&2 <<ERRMSG
ERROR: Failed to resolve a JAX params directory from BASE_MODEL_PATH=${BASE_MODEL_PATH}

Supported layouts:
  1. /path/to/checkpoint/params
  2. /path/to/checkpoint_root/<step>/params

You can also set BASE_CKPT_STEP explicitly, for example:
  BASE_CKPT_STEP=5000 bash scripts/train_scripts/train_jax.sh
ERRMSG
    exit 1
  fi
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "ERROR: Virtualenv not found at ${VENV_PATH}" >&2
  exit 1
fi

source "${VENV_PATH}/bin/activate"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/.pydeps:${PROJECT_DIR}/src:${PROJECT_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_NVLS_ENABLE=0

DETECTED_JAX_DEVICES="$(python -c 'import jax; print(jax.device_count())' 2>/dev/null || true)"
if [[ -n "${DETECTED_JAX_DEVICES}" && "${DETECTED_JAX_DEVICES}" != "${FSDP_DEVICES}" ]]; then
  cat >&2 <<ERRMSG
ERROR: JAX sees ${DETECTED_JAX_DEVICES} devices, but FSDP_DEVICES=${FSDP_DEVICES}.

Current settings:
  VENV_PATH=${VENV_PATH}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

Please verify the shared server environment and your CUDA device selection.
ERRMSG
  exit 1
fi

if [[ "${GRAD_ACCUM_STEPS}" != "1" ]]; then
  echo "WARNING: scripts/train.py currently does not use gradient_accumulation_steps; only BATCH_SIZE takes effect." >&2
fi

CMD=(
  python scripts/train.py "${CONFIG_NAME}"
  --exp_name "${EXP_NAME}"
  --checkpoint_base_dir "${CHECKPOINT_BASE_DIR}"
  --data.default_prompt "${DEFAULT_PROMPT}"
  --data.data_root "${DATA_ROOT}"
  --data.assets.assets_dir "${DATA_ROOT}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --num_train_steps "${NUM_TRAIN_STEPS}"
  --save_interval "${SAVE_INTERVAL}"
  --keep_period "${KEEP_PERIOD}"
  --lr_schedule.peak_lr "${PEAK_LR}"
  --lr_schedule.decay_lr "${DECAY_LR}"
  --fsdp_devices "${FSDP_DEVICES}"
)

if [[ "${RESUME}" != "1" ]]; then
  CMD+=(--weight_loader.params_path "${WEIGHT_LOADER_PATH}")
fi

if [[ "${WANDB_ENABLED}" == "1" ]]; then
  CMD+=(--wandb_enabled)
else
  CMD+=(--no_wandb_enabled)
fi

if [[ "${RESUME}" == "1" ]]; then
  CMD+=(--resume)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}"
if [[ "${RESUME}" == "1" ]]; then
  echo "WEIGHT_LOADER_PATH=<resume>"
else
  echo "WEIGHT_LOADER_PATH=${WEIGHT_LOADER_PATH}"
fi
echo "FSDP_DEVICES=${FSDP_DEVICES}"

"${CMD[@]}"
