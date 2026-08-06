#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/../ckpts/Wan21/Action2V/dmd/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
DURATIONS="${DURATIONS:-10s:40 15s:60 20s:80 30s:120}"
PROMPT_START="${PROMPT_START:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
SEEDS="${SEEDS:-0}"
SINK_SIZE="${SINK_SIZE:-4}"
RETRIEVAL_FRAMES="${RETRIEVAL_FRAMES:-8}"
RETRIEVAL_RECENT_FRAMES="${RETRIEVAL_RECENT_FRAMES:-8}"
ROPE_TRAIN_LENGTH="${ROPE_TRAIN_LENGTH:-19}"
ROPE_LOCAL_WINDOW="${ROPE_LOCAL_WINDOW:-4}"
CASES="${CASES:-fixed_sink fixed_sink_only_rope_rebase tri_region_rope_rebase}"

if [ -z "$CUDA_DEVICES" ] || [[ "$CUDA_DEVICES" =~ [[:space:]] ]]; then
  echo "CUDA_DEVICES must be a non-empty comma-separated list without spaces." >&2
  exit 2
fi

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_DEVICES"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-${#CUDA_DEVICE_LIST[@]}}"
SP_SIZE="${SP_SIZE:-$NUM_GPUS_PER_NODE}"
if [ "$NUM_GPUS_PER_NODE" -ne "${#CUDA_DEVICE_LIST[@]}" ] || [ "$SP_SIZE" -ne "$NUM_GPUS_PER_NODE" ]; then
  echo "NUM_GPUS_PER_NODE and SP_SIZE must match the number of CUDA_DEVICES." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"

for spec in $DURATIONS; do
  label="${spec%%:*}"
  frames="${spec##*:}"
  trajectory_path="Wan21/prompts/demos_loop_closure/trajectories_${label}.txt"
  run_root="$OUTPUT_ROOT/string_loop_${label}_rope_rebase_cases_seed0"

  if [ ! -f "$trajectory_path" ]; then
    echo "Missing trajectory file: $trajectory_path" >&2
    exit 2
  fi

  echo "===== duration=$label latent_frames=$frames cases=$CASES ====="
  env \
    PYTHONPATH="$PROJECT_ROOT/Wan21:$PROJECT_ROOT/shared:${PYTHONPATH:-}" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_ROOT="$run_root" \
    CONFIG_PATH=Wan21/configs/causal_forcing_dmd_camera.yaml \
    CHECKPOINT_PATH="$CHECKPOINT_PATH" \
    DATA_PATH=Wan21/prompts/demos_loop_closure/prompts.txt \
    TRAJECTORY_PATH="$trajectory_path" \
    NUM_OUTPUT_FRAMES="$frames" \
    PROMPT_START="$PROMPT_START" \
    MAX_PROMPTS="$MAX_PROMPTS" \
    SEEDS="$SEEDS" \
    CASES="$CASES" \
    NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" \
    SP_SIZE="$SP_SIZE" \
    SINK_SIZE="$SINK_SIZE" \
    KV_BANK_DEVICE=cpu \
    KV_BANK_MAX_BLOCKS=10 \
    KV_BANK_WARN_MEMORY_GB=128 \
    RETRIEVAL_FRAMES="$RETRIEVAL_FRAMES" \
    RETRIEVAL_RECENT_FRAMES="$RETRIEVAL_RECENT_FRAMES" \
    RETRIEVAL_FOV_SAMPLES=8192 \
    RETRIEVAL_FOV_RADIUS=8.0 \
    RETRIEVAL_FOV_H_DEG=60.0 \
    RETRIEVAL_FOV_V_DEG=35.0 \
    RETRIEVAL_ROPE_CORRECTION=0 \
    ROPE_TRAIN_LENGTH="$ROPE_TRAIN_LENGTH" \
    ROPE_LOCAL_WINDOW="$ROPE_LOCAL_WINDOW" \
    KV_COMPRESSION_KEEP_RATIO=0.5 \
    EVAL_LOOP_CLOSURE_ENABLE=1 \
    EVAL_LOOP_MANIFEST=Wan21/prompts/demos_loop_closure/manifest.json \
    EVAL_LOOP_DURATION_LABEL="$label" \
    EVAL_STYLE_ENABLE=1 \
    EVAL_OFFICIAL_VBENCH_ENABLE=0 \
    SKIP_COMPLETED=1 \
    CONTINUE_ON_ERROR=0 \
    bash "$SCRIPT_DIR/run_string_camera_experiments.sh"
done
