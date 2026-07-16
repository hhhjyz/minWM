#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

# Smoke-test defaults. Override any value from the command line environment.
export DATA_PATH="${DATA_PATH:-Wan21/prompts/tartanair_long5_120/prompts.txt}"
export TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/tartanair_long5_120/trajectories.txt}"
export MAX_PROMPTS="${MAX_PROMPTS:-1}"
export PROMPT_START="${PROMPT_START:-0}"
export NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-120}"
export LOG_CACHE_STATE="${LOG_CACHE_STATE:-1}"
export LOG_CACHE_INTERVAL="${LOG_CACHE_INTERVAL:-1}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/pool/hdd/home/hhhjyz/research/ckpts/Wan21/Action2V/dmd/model.pt}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs/tartanair_long5_120_${PROMPT_START}_${MAX_PROMPTS}_${NUM_OUTPUT_FRAMES}}"

bash "$SCRIPT_DIR/run_infer_causal_camera.sh"
