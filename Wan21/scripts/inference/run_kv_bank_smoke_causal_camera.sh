#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

export DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
export TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/trajectories.txt}"
export MAX_PROMPTS="${MAX_PROMPTS:-1}"
export PROMPT_START="${PROMPT_START:-0}"
export NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
export KV_BANK_ENABLE="${KV_BANK_ENABLE:-1}"
export KV_BANK_DEVICE="${KV_BANK_DEVICE:-cpu}"
export KV_BANK_MAX_BLOCKS="${KV_BANK_MAX_BLOCKS:-2}"
export KV_BANK_LOG_INTERVAL="${KV_BANK_LOG_INTERVAL:-1}"
export KV_BANK_WARN_MEMORY_GB="${KV_BANK_WARN_MEMORY_GB:-8}"
export LOG_CACHE_STATE="${LOG_CACHE_STATE:-0}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs/kv_bank_smoke_${PROMPT_START}_${MAX_PROMPTS}_${NUM_OUTPUT_FRAMES}}"

bash "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"
