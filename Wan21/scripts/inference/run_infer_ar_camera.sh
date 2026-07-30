#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

export NCCL_DEBUG=WARN

# ===== Paths =====
CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/ar_camera_tf.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/ar_camera}"
SP_SIZE="${SP_SIZE:-1}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
SEED="${SEED:-0}"
MAX_PROMPTS="${MAX_PROMPTS:--1}"
PROMPT_START="${PROMPT_START:-0}"
LOG_CACHE_STATE="${LOG_CACHE_STATE:-0}"
LOG_CACHE_INTERVAL="${LOG_CACHE_INTERVAL:-1}"
SINK_STRATEGY="${SINK_STRATEGY:-none}"
SINK_SIZE="${SINK_SIZE:-0}"
FIXED_SINK_ROPE_REBASE="${FIXED_SINK_ROPE_REBASE:-0}"
TRI_REGION_ROPE_REBASE="${TRI_REGION_ROPE_REBASE:-$FIXED_SINK_ROPE_REBASE}"
ROPE_TRAIN_LENGTH="${ROPE_TRAIN_LENGTH:-21}"
ROPE_LOCAL_WINDOW="${ROPE_LOCAL_WINDOW:-9}"
SINK_UPDATE_INTERVAL="${SINK_UPDATE_INTERVAL:-0}"
SINK_BANK_SEED="${SINK_BANK_SEED:-0}"
PROPE_REENCODE_MODE="${PROPE_REENCODE_MODE:-none}"

# ===== Camera Trajectory =====
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

# Build trajectory argument
if [ -n "$TRAJECTORY_PATH" ]; then
  TRAJ_ARGS="--trajectory_path $TRAJECTORY_PATH"
else
  TRAJ_ARGS="--trajectory $TRAJECTORY"
fi

LOG_ARGS=""
if [ "$LOG_CACHE_STATE" = "1" ] || [ "$LOG_CACHE_STATE" = "true" ] || [ "$LOG_CACHE_STATE" = "True" ]; then
  LOG_ARGS="--log_cache_state --log_cache_interval $LOG_CACHE_INTERVAL"
fi

SINK_REBASE_ARGS="--rope_train_length $ROPE_TRAIN_LENGTH --rope_local_window $ROPE_LOCAL_WINDOW"
if [ "$TRI_REGION_ROPE_REBASE" = "1" ] || [ "$TRI_REGION_ROPE_REBASE" = "true" ] || [ "$TRI_REGION_ROPE_REBASE" = "True" ]; then
  SINK_REBASE_ARGS="--tri_region_rope_rebase $SINK_REBASE_ARGS"
fi

NUM_GPUS_PER_NODE=1
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29621}

echo "=== Inference: AR Camera Control ==="
echo "  Config:     $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $OUTPUT_FOLDER"
echo "  Subset:     start=$PROMPT_START max=$MAX_PROMPTS frames=$NUM_OUTPUT_FRAMES seed=$SEED"
echo "  Trajectory: ${TRAJECTORY_PATH:-$TRAJECTORY}"
echo "  Sink:       strategy=$SINK_STRATEGY size=$SINK_SIZE update_interval=$SINK_UPDATE_INTERVAL bank_seed=$SINK_BANK_SEED"
echo "  Tri RoPE:   enabled=$TRI_REGION_ROPE_REBASE train_length=$ROPE_TRAIN_LENGTH local_window=$ROPE_LOCAL_WINDOW"
echo "  PRoPE:      reencode_mode=$PROPE_REENCODE_MODE"

export SP_SIZE=$SP_SIZE
torchrun \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  Wan21/wan_inference.py \
  --config_path "$CONFIG_PATH" \
  --output_folder "$OUTPUT_FOLDER" \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --data_path "$DATA_PATH" \
  --num_output_frames "$NUM_OUTPUT_FRAMES" \
  --seed "$SEED" \
  --max_prompts "$MAX_PROMPTS" \
  --prompt_start "$PROMPT_START" \
  --sink_strategy "$SINK_STRATEGY" \
  --sink_size "$SINK_SIZE" \
  --sink_update_interval "$SINK_UPDATE_INTERVAL" \
  --sink_bank_seed "$SINK_BANK_SEED" \
  $SINK_REBASE_ARGS \
  --sp_size $SP_SIZE \
  --prope_reencode_mode "$PROPE_REENCODE_MODE" \
  $LOG_ARGS \
  $TRAJ_ARGS
