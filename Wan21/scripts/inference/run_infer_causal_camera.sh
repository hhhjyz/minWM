#!/bin/bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

export NCCL_DEBUG=WARN

# ===== Paths =====
# Default = Stage 3 (DMD) checkpoint, matching §3.3 quickstart.
# Override CONFIG_PATH + CHECKPOINT_PATH together to validate Stage 2(a)/(b):
#   Stage 2(a) ODE: CONFIG_PATH=Wan21/configs/causal_ode_camera.yaml
#                   CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_ode/model.pt
#   Stage 2(b) CD:  CONFIG_PATH=Wan21/configs/causal_cd_camera.yaml
#                   CHECKPOINT_PATH=./ckpts/Wan21/Action2V/causal_cd/model.pt
CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-../ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/causal_camera}"
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
KV_BANK_ENABLE="${KV_BANK_ENABLE:-0}"
KV_BANK_DEVICE="${KV_BANK_DEVICE:-cpu}"
KV_BANK_MAX_BLOCKS="${KV_BANK_MAX_BLOCKS:-0}"
KV_BANK_LOG_INTERVAL="${KV_BANK_LOG_INTERVAL:-1}"
KV_BANK_WARN_MEMORY_GB="${KV_BANK_WARN_MEMORY_GB:-16}"
RETRIEVAL_ENABLE="${RETRIEVAL_ENABLE:-0}"
RETRIEVAL_GRANULARITY="${RETRIEVAL_GRANULARITY:-chunk}"
RETRIEVAL_METRIC="${RETRIEVAL_METRIC:-pose}"
RETRIEVAL_FRAMES="${RETRIEVAL_FRAMES:-0}"
RETRIEVAL_RECENT_FRAMES="${RETRIEVAL_RECENT_FRAMES:-0}"
RETRIEVAL_FOV_SAMPLES="${RETRIEVAL_FOV_SAMPLES:-8192}"
RETRIEVAL_FOV_RADIUS="${RETRIEVAL_FOV_RADIUS:-8.0}"
RETRIEVAL_FOV_H_DEG="${RETRIEVAL_FOV_H_DEG:-60.0}"
RETRIEVAL_FOV_V_DEG="${RETRIEVAL_FOV_V_DEG:-35.0}"
RETRIEVAL_HYBRID_FOV_WEIGHT="${RETRIEVAL_HYBRID_FOV_WEIGHT:-0.5}"
RETRIEVAL_ROPE_CORRECTION="${RETRIEVAL_ROPE_CORRECTION:-0}"
PROPE_REENCODE_MODE="${PROPE_REENCODE_MODE:-none}"
KV_COMPRESSION_ENABLE="${KV_COMPRESSION_ENABLE:-0}"
KV_COMPRESSION_KEEP_RATIO="${KV_COMPRESSION_KEEP_RATIO:-0.5}"
KV_COMPRESSION_ANCHOR_ROTATE="${KV_COMPRESSION_ANCHOR_ROTATE:-0}"
KV_COMPRESSION_AT_STORE="${KV_COMPRESSION_AT_STORE:-0}"
KV_COMPRESSION_POOLED="${KV_COMPRESSION_POOLED:-0}"
KV_COMPRESSION_DYNAMIC_ENABLE="${KV_COMPRESSION_DYNAMIC_ENABLE:-0}"
KV_COMPRESSION_DYNAMIC_MIN_KEEP="${KV_COMPRESSION_DYNAMIC_MIN_KEEP:-0.25}"
KV_COMPRESSION_DYNAMIC_MAX_KEEP="${KV_COMPRESSION_DYNAMIC_MAX_KEEP:-0.75}"
KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE="${KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE:-1.0}"
KV_COMPRESSION_DYNAMIC_ROTATION_SCALE="${KV_COMPRESSION_DYNAMIC_ROTATION_SCALE:-0.35}"
KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT="${KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT:-0.25}"

# ===== Camera Trajectory =====
TRAJECTORY="${TRAJECTORY:-w*19}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

if [ -n "$TRAJECTORY_PATH" ]; then
  TRAJ_ARGS="--trajectory_path $TRAJECTORY_PATH"
else
  TRAJ_ARGS="--trajectory $TRAJECTORY"
fi

LOG_ARGS=""
if [ "$LOG_CACHE_STATE" = "1" ] || [ "$LOG_CACHE_STATE" = "true" ] || [ "$LOG_CACHE_STATE" = "True" ]; then
  LOG_ARGS="--log_cache_state --log_cache_interval $LOG_CACHE_INTERVAL"
fi

KV_BANK_ARGS="--kv_bank_device $KV_BANK_DEVICE --kv_bank_max_blocks $KV_BANK_MAX_BLOCKS --kv_bank_log_interval $KV_BANK_LOG_INTERVAL --kv_bank_warn_memory_gb $KV_BANK_WARN_MEMORY_GB"
if [ "$KV_BANK_ENABLE" = "1" ] || [ "$KV_BANK_ENABLE" = "true" ] || [ "$KV_BANK_ENABLE" = "True" ]; then
  KV_BANK_ARGS="--kv_bank_enable $KV_BANK_ARGS"
fi

RETRIEVAL_ARGS="--retrieval_granularity $RETRIEVAL_GRANULARITY --retrieval_metric $RETRIEVAL_METRIC --retrieval_frames $RETRIEVAL_FRAMES --retrieval_recent_frames $RETRIEVAL_RECENT_FRAMES --retrieval_fov_samples $RETRIEVAL_FOV_SAMPLES --retrieval_fov_radius $RETRIEVAL_FOV_RADIUS --retrieval_fov_h_deg $RETRIEVAL_FOV_H_DEG --retrieval_fov_v_deg $RETRIEVAL_FOV_V_DEG --retrieval_hybrid_fov_weight $RETRIEVAL_HYBRID_FOV_WEIGHT"
if [ "$RETRIEVAL_ENABLE" = "1" ] || [ "$RETRIEVAL_ENABLE" = "true" ] || [ "$RETRIEVAL_ENABLE" = "True" ]; then
  RETRIEVAL_ARGS="--retrieval_enable $RETRIEVAL_ARGS"
fi
if [ "$RETRIEVAL_ROPE_CORRECTION" = "1" ] || [ "$RETRIEVAL_ROPE_CORRECTION" = "true" ] || [ "$RETRIEVAL_ROPE_CORRECTION" = "True" ]; then
  RETRIEVAL_ARGS="$RETRIEVAL_ARGS --retrieval_rope_correction"
fi

SINK_REBASE_ARGS="--rope_train_length $ROPE_TRAIN_LENGTH --rope_local_window $ROPE_LOCAL_WINDOW"
if [ "$TRI_REGION_ROPE_REBASE" = "1" ] || [ "$TRI_REGION_ROPE_REBASE" = "true" ] || [ "$TRI_REGION_ROPE_REBASE" = "True" ]; then
  SINK_REBASE_ARGS="--tri_region_rope_rebase $SINK_REBASE_ARGS"
fi

COMPRESSION_ARGS="--kv_compression_keep_ratio $KV_COMPRESSION_KEEP_RATIO"
if [ "$KV_COMPRESSION_ENABLE" = "1" ] || [ "$KV_COMPRESSION_ENABLE" = "true" ] || [ "$KV_COMPRESSION_ENABLE" = "True" ]; then
  COMPRESSION_ARGS="--kv_compression_enable $COMPRESSION_ARGS"
fi
if [ "$KV_COMPRESSION_ANCHOR_ROTATE" = "1" ] || [ "$KV_COMPRESSION_ANCHOR_ROTATE" = "true" ] || [ "$KV_COMPRESSION_ANCHOR_ROTATE" = "True" ]; then
  COMPRESSION_ARGS="$COMPRESSION_ARGS --kv_compression_anchor_rotate"
fi
if [ "$KV_COMPRESSION_AT_STORE" = "1" ] || [ "$KV_COMPRESSION_AT_STORE" = "true" ] || [ "$KV_COMPRESSION_AT_STORE" = "True" ]; then
  COMPRESSION_ARGS="$COMPRESSION_ARGS --kv_compression_at_store"
fi
if [ "$KV_COMPRESSION_POOLED" = "1" ] || [ "$KV_COMPRESSION_POOLED" = "true" ] || [ "$KV_COMPRESSION_POOLED" = "True" ]; then
  COMPRESSION_ARGS="$COMPRESSION_ARGS --kv_compression_pooled"
fi
if [ "$KV_COMPRESSION_DYNAMIC_ENABLE" = "1" ] || [ "$KV_COMPRESSION_DYNAMIC_ENABLE" = "true" ] || [ "$KV_COMPRESSION_DYNAMIC_ENABLE" = "True" ]; then
  COMPRESSION_ARGS="$COMPRESSION_ARGS --kv_compression_dynamic_enable"
fi
COMPRESSION_ARGS="$COMPRESSION_ARGS --kv_compression_dynamic_min_keep $KV_COMPRESSION_DYNAMIC_MIN_KEEP --kv_compression_dynamic_max_keep $KV_COMPRESSION_DYNAMIC_MAX_KEEP --kv_compression_dynamic_translation_scale $KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE --kv_compression_dynamic_rotation_scale $KV_COMPRESSION_DYNAMIC_ROTATION_SCALE --kv_compression_dynamic_motion_weight $KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT"

NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-1}"
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29622}

echo "=== Inference: Causal Camera Control (ODE/CD/DMD) ==="
echo "  Config:     $CONFIG_PATH"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $OUTPUT_FOLDER"
echo "  Subset:     start=$PROMPT_START max=$MAX_PROMPTS frames=$NUM_OUTPUT_FRAMES seed=$SEED"
echo "  Trajectory: ${TRAJECTORY_PATH:-$TRAJECTORY}"
echo "  Sink:       strategy=$SINK_STRATEGY size=$SINK_SIZE update_interval=$SINK_UPDATE_INTERVAL bank_seed=$SINK_BANK_SEED"
echo "  KV bank:    enabled=$KV_BANK_ENABLE device=$KV_BANK_DEVICE max_blocks=$KV_BANK_MAX_BLOCKS"
echo "  Retrieval:  enabled=$RETRIEVAL_ENABLE metric=$RETRIEVAL_METRIC frames=$RETRIEVAL_FRAMES recent_exclusion=$RETRIEVAL_RECENT_FRAMES rope_correction=$RETRIEVAL_ROPE_CORRECTION"
echo "  Tri RoPE:   enabled=$TRI_REGION_ROPE_REBASE train_length=$ROPE_TRAIN_LENGTH local_window=$ROPE_LOCAL_WINDOW"
echo "  PRoPE:      reencode_mode=$PROPE_REENCODE_MODE"
echo "  Compression enabled=$KV_COMPRESSION_ENABLE keep_ratio=$KV_COMPRESSION_KEEP_RATIO at_store=$KV_COMPRESSION_AT_STORE pooled=$KV_COMPRESSION_POOLED"
echo "  Dynamic compression enabled=$KV_COMPRESSION_DYNAMIC_ENABLE min=$KV_COMPRESSION_DYNAMIC_MIN_KEEP max=$KV_COMPRESSION_DYNAMIC_MAX_KEEP trans_scale=$KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE rot_scale=$KV_COMPRESSION_DYNAMIC_ROTATION_SCALE motion_weight=$KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT"

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
  $KV_BANK_ARGS \
  $RETRIEVAL_ARGS \
  $COMPRESSION_ARGS \
  $TRAJ_ARGS
