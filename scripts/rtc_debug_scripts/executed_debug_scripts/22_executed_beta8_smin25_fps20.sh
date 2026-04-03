#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common_experiment_lib.sh"

rtc_experiment_init "$0"
FPS="${FPS:-20}"
RTC_BETA="${RTC_BETA:-8.0}"
RTC_S_MIN="${RTC_S_MIN:-25}"

run_jax_rtc_experiment executed_overlap_paper
