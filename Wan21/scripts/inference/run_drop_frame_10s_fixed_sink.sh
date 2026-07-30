#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

export DATA_PATH="${DATA_PATH:-Wan21/prompts/drop_frame_10s/prompts.txt}"
export TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/drop_frame_10s/trajectories.txt}"
export CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-../ckpts/Wan21/Action2V/dmd/model.pt}"
export NUM_OUTPUT_FRAMES=40
export PROMPT_START="${PROMPT_START:-0}"
export MAX_PROMPTS="${MAX_PROMPTS:-30}"
export SEED="${SEED:-0}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs/drop_frame_10s_fixed_sink_seed${SEED}}"

# This stress set intentionally tests fixed sink without sidecar memory.
export SINK_STRATEGY=fixed
export SINK_SIZE="${SINK_SIZE:-4}"
export SINK_UPDATE_INTERVAL=0
export KV_BANK_ENABLE=0
export RETRIEVAL_ENABLE=0
export RETRIEVAL_FRAMES=0
export KV_COMPRESSION_ENABLE=0
export KV_COMPRESSION_AT_STORE=0

exec bash "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"
