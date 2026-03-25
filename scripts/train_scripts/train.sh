#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-/jedata/jemotor/code/openpi/.venv}"
PROJECT_DIR="${PROJECT_DIR:-/jedata/jemotor/code/openpi_pi05_torch_1}"
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
s
source "${VENV_PATH}/bin/activate"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/.pydeps:${PROJECT_DIR}/src:${PROJECT_DIR}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

NCCL_NVLS_ENABLE=0 CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/train_pytorch.py pi05_agileX_thor \
  --exp_name "${EXP_NAME}" \
  --data.default_prompt "${DEFAULT_PROMPT}" \
  --data.data_root "${DATA_ROOT}" \
  --data.assets.assets_dir "${DATA_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --num_workers "${NUM_WORKERS}" \
  --num_train_steps "${NUM_TRAIN_STEPS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --keep_period "${KEEP_PERIOD}" \
  --lr_schedule.peak_lr "${PEAK_LR}" \
  --lr_schedule.decay_lr "${DECAY_LR}" \
  --no_wandb_enabled
