#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"

DATA_PATH="${DATA_PATH:-Wan21/prompts/tartanair_long5_120/prompts.txt}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/tartanair_long5_120/trajectories.txt}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
PROMPT_START="${PROMPT_START:-0}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-120}"
SINK_SIZE="${SINK_SIZE:-4}"
SINK_UPDATE_INTERVAL="${SINK_UPDATE_INTERVAL:-4}"
RUN_PREFIX="${RUN_PREFIX:-sink_ablation}"

run_case() {
  local name="$1"
  local strategy="$2"
  local sink_size="$3"
  local interval="$4"

  DATA_PATH="$DATA_PATH" \
  TRAJECTORY_PATH="$TRAJECTORY_PATH" \
  MAX_PROMPTS="$MAX_PROMPTS" \
  PROMPT_START="$PROMPT_START" \
  NUM_OUTPUT_FRAMES="$NUM_OUTPUT_FRAMES" \
  SINK_STRATEGY="$strategy" \
  SINK_SIZE="$sink_size" \
  SINK_UPDATE_INTERVAL="$interval" \
  OUTPUT_FOLDER="./outputs/${RUN_PREFIX}_${name}_${PROMPT_START}_${MAX_PROMPTS}_${NUM_OUTPUT_FRAMES}" \
  bash "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"
}

run_case "baseline" "none" "0" "0"
run_case "fixed_sink${SINK_SIZE}" "fixed" "$SINK_SIZE" "0"
run_case "periodic_sink${SINK_SIZE}_int${SINK_UPDATE_INTERVAL}" "periodic" "$SINK_SIZE" "$SINK_UPDATE_INTERVAL"
