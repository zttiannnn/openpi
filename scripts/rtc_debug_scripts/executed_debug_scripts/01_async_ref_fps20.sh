#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_experiment_lib.sh"

rtc_experiment_init "$0"
FPS="${FPS:-20}"
ACTION_STEPS="${ACTION_STEPS:-20}"
PLOT_CHUNK_HANDOFF_ENABLE="${PLOT_CHUNK_HANDOFF_ENABLE:-0}"

run_async_experiment
