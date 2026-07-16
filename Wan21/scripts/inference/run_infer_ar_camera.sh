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
MAX_PROMPTS="${MAX_PROMPTS:--1}"
PROMPT_START="${PROMPT_START:-0}"
LOG_CACHE_STATE="${LOG_CACHE_STATE:-0}"
LOG_CACHE_INTERVAL="${LOG_CACHE_INTERVAL:-1}"
SINK_STRATEGY="${SINK_STRATEGY:-none}"
SINK_SIZE="${SINK_SIZE:-0}"
SINK_UPDATE_INTERVAL="${SINK_UPDATE_INTERVAL:-0}"

# ===== Camera Trajectory =====
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"
TRAJECTORY_POSE_PATH="${TRAJECTORY_POSE_PATH:-}"

# Build trajectory argument
if [ -n "$TRAJECTORY_POSE_PATH" ]; then
  TRAJ_ARGS="--trajectory_pose_path $TRAJECTORY_POSE_PATH"
elif [ -n "$TRAJECTORY_PATH" ]; then
  TRAJ_ARGS="--trajectory_path $TRAJECTORY_PATH"
else
  TRAJ_ARGS="--trajectory $TRAJECTORY"
fi

LOG_ARGS=""
if [ "$LOG_CACHE_STATE" = "1" ] || [ "$LOG_CACHE_STATE" = "true" ] || [ "$LOG_CACHE_STATE" = "True" ]; then
  LOG_ARGS="--log_cache_state --log_cache_interval $LOG_CACHE_INTERVAL"
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
echo "  Subset:     start=$PROMPT_START max=$MAX_PROMPTS frames=$NUM_OUTPUT_FRAMES"
echo "  Pose path:  ${TRAJECTORY_POSE_PATH:-<trajectory-string-mode>}"
echo "  Sink:       strategy=$SINK_STRATEGY size=$SINK_SIZE update_interval=$SINK_UPDATE_INTERVAL"

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
  --max_prompts "$MAX_PROMPTS" \
  --prompt_start "$PROMPT_START" \
  --sink_strategy "$SINK_STRATEGY" \
  --sink_size "$SINK_SIZE" \
  --sink_update_interval "$SINK_UPDATE_INTERVAL" \
  --sp_size $SP_SIZE \
  $LOG_ARGS \
  $TRAJ_ARGS
