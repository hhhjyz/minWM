#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

RUN_ROOT="${RUN_ROOT:-./outputs/string_camera_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-../ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
TRAJECTORY_PATH="${TRAJECTORY_PATH-Wan21/prompts/trajectories.txt}"
TRAJECTORY="${TRAJECTORY:-}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
MAX_PROMPTS="${MAX_PROMPTS:-5}"
PROMPT_START="${PROMPT_START:-0}"
SEEDS="${SEEDS:-0}"
SP_SIZE="${SP_SIZE:-1}"

SINK_SIZE="${SINK_SIZE:-4}"
SINK_UPDATE_INTERVAL="${SINK_UPDATE_INTERVAL:-8}"
SINK_BANK_SEED="${SINK_BANK_SEED:-0}"

KV_BANK_DEVICE="${KV_BANK_DEVICE:-cpu}"
KV_BANK_MAX_BLOCKS="${KV_BANK_MAX_BLOCKS:-45}"
KV_BANK_LOG_INTERVAL="${KV_BANK_LOG_INTERVAL:-1}"
KV_BANK_WARN_MEMORY_GB="${KV_BANK_WARN_MEMORY_GB:-128}"

RETRIEVAL_FRAMES="${RETRIEVAL_FRAMES:-12}"
RETRIEVAL_RECENT_FRAMES="${RETRIEVAL_RECENT_FRAMES:-8}"
RETRIEVAL_FOV_SAMPLES="${RETRIEVAL_FOV_SAMPLES:-8192}"
RETRIEVAL_FOV_RADIUS="${RETRIEVAL_FOV_RADIUS:-8.0}"
RETRIEVAL_FOV_H_DEG="${RETRIEVAL_FOV_H_DEG:-60.0}"
RETRIEVAL_FOV_V_DEG="${RETRIEVAL_FOV_V_DEG:-35.0}"
RETRIEVAL_HYBRID_FOV_WEIGHT="${RETRIEVAL_HYBRID_FOV_WEIGHT:-0.5}"
RETRIEVAL_ROPE_CORRECTION="${RETRIEVAL_ROPE_CORRECTION:-0}"
PROPE_REENCODE_MODE="${PROPE_REENCODE_MODE:-none}"

KV_COMPRESSION_KEEP_RATIO="${KV_COMPRESSION_KEEP_RATIO:-0.5}"
KV_COMPRESSION_DYNAMIC_MIN_KEEP="${KV_COMPRESSION_DYNAMIC_MIN_KEEP:-0.25}"
KV_COMPRESSION_DYNAMIC_MAX_KEEP="${KV_COMPRESSION_DYNAMIC_MAX_KEEP:-0.75}"
KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE="${KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE:-1.0}"
KV_COMPRESSION_DYNAMIC_ROTATION_SCALE="${KV_COMPRESSION_DYNAMIC_ROTATION_SCALE:-0.35}"
KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT="${KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT:-0.25}"

LOG_CACHE_STATE="${LOG_CACHE_STATE:-0}"
LOG_CACHE_INTERVAL="${LOG_CACHE_INTERVAL:-1}"
EVAL_STYLE_ENABLE="${EVAL_STYLE_ENABLE:-1}"
EVAL_STYLE_MAX_FRAMES="${EVAL_STYLE_MAX_FRAMES:-96}"
EVAL_STYLE_RESIZE_WIDTH="${EVAL_STYLE_RESIZE_WIDTH:-256}"
EVAL_LOOP_CLOSURE_ENABLE="${EVAL_LOOP_CLOSURE_ENABLE:-auto}"
EVAL_LOOP_MANIFEST="${EVAL_LOOP_MANIFEST:-Wan21/prompts/demos_loop_closure/manifest.json}"
EVAL_LOOP_DURATION_LABEL="${EVAL_LOOP_DURATION_LABEL:-}"
EVAL_LOOP_MAX_FRAMES="${EVAL_LOOP_MAX_FRAMES:-96}"
EVAL_LOOP_RESIZE_WIDTH="${EVAL_LOOP_RESIZE_WIDTH:-256}"
EVAL_LOOP_DEVICE="${EVAL_LOOP_DEVICE:-auto}"
EVAL_LOOP_LPIPS_BATCH_SIZE="${EVAL_LOOP_LPIPS_BATCH_SIZE:-64}"
EVAL_LOOP_SKIP_LPIPS="${EVAL_LOOP_SKIP_LPIPS:-0}"
EVAL_OFFICIAL_VBENCH_ENABLE="${EVAL_OFFICIAL_VBENCH_ENABLE:-0}"
OFFICIAL_VBENCH_ROOT="${OFFICIAL_VBENCH_ROOT:-../VBench}"
OFFICIAL_VBENCH_PYTHON="${OFFICIAL_VBENCH_PYTHON:-/home/hhhjyz/miniconda3/envs/vbench/bin/python}"
OFFICIAL_VBENCH_DIMENSIONS="${OFFICIAL_VBENCH_DIMENSIONS:-subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality}"
OFFICIAL_VBENCH_LOAD_LOCAL="${OFFICIAL_VBENCH_LOAD_LOCAL:-1}"
OFFICIAL_VBENCH_CACHE_DIR="${OFFICIAL_VBENCH_CACHE_DIR:-$OFFICIAL_VBENCH_ROOT/pretrained/cache}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_CASES=(
  baseline
  fixed_sink
  periodic_sink
  pose_compress_store
  worldkv_fov_compress_store
  worldkv_fov_dynamic_compress_store
)
if [ -n "${CASES:-}" ]; then
  # shellcheck disable=SC2206
  CASE_LIST=($CASES)
else
  CASE_LIST=("${DEFAULT_CASES[@]}")
fi

bool_enabled() {
  case "$1" in
    1|true|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

loop_closure_eval_enabled() {
  case "$EVAL_LOOP_CLOSURE_ENABLE" in
    auto|AUTO|Auto)
      [[ "$TRAJECTORY_PATH" == *"/demos_loop_closure/trajectories_"*".txt" ]]
      ;;
    *) bool_enabled "$EVAL_LOOP_CLOSURE_ENABLE" ;;
  esac
}

if [ -n "$TRAJECTORY" ]; then
  TRAJECTORY_PATH=""
fi
if [ -z "$TRAJECTORY" ] && [ -z "$TRAJECTORY_PATH" ]; then
  echo "Set TRAJECTORY or TRAJECTORY_PATH." >&2
  exit 2
fi
if [ ! -f "$DATA_PATH" ]; then
  echo "Prompt file does not exist: $DATA_PATH" >&2
  exit 2
fi
if [ -n "$TRAJECTORY_PATH" ] && [ ! -f "$TRAJECTORY_PATH" ]; then
  echo "Trajectory file does not exist: $TRAJECTORY_PATH" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
MANIFEST="$RUN_ROOT/experiment_manifest.txt"
SUMMARY="$RUN_ROOT/experiment_summary.tsv"

cat > "$MANIFEST" <<EOF
run_root=$RUN_ROOT
config_path=$CONFIG_PATH
checkpoint_path=$CHECKPOINT_PATH
data_path=$DATA_PATH
trajectory=${TRAJECTORY:-}
trajectory_path=${TRAJECTORY_PATH:-}
num_output_frames=$NUM_OUTPUT_FRAMES
max_prompts=$MAX_PROMPTS
prompt_start=$PROMPT_START
seeds=$SEEDS
cases=${CASE_LIST[*]}
sink_size=$SINK_SIZE
sink_update_interval=$SINK_UPDATE_INTERVAL
kv_bank_device=$KV_BANK_DEVICE
kv_bank_max_blocks=$KV_BANK_MAX_BLOCKS
retrieval_frames=$RETRIEVAL_FRAMES
retrieval_granularity_cases=chunk,latent_frame
retrieval_recent_frames=$RETRIEVAL_RECENT_FRAMES
retrieval_fov_samples=$RETRIEVAL_FOV_SAMPLES
retrieval_fov_radius=$RETRIEVAL_FOV_RADIUS
retrieval_fov_h_deg=$RETRIEVAL_FOV_H_DEG
retrieval_fov_v_deg=$RETRIEVAL_FOV_V_DEG
prope_reencode_mode=$PROPE_REENCODE_MODE
compression_keep_ratio=$KV_COMPRESSION_KEEP_RATIO
official_vbench=$EVAL_OFFICIAL_VBENCH_ENABLE
loop_closure_eval=$EVAL_LOOP_CLOSURE_ENABLE
loop_closure_manifest=$EVAL_LOOP_MANIFEST
loop_closure_max_frames=$EVAL_LOOP_MAX_FRAMES
EOF
printf "seed\tcase\tinference\tevaluation\toutput_folder\n" > "$SUMMARY"

pick_master_port() {
  "$PYTHON_BIN" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
}

inference_complete() {
  local timing_csv="$1/inference_times.csv"
  [ -f "$timing_csv" ] || return 1
  "$PYTHON_BIN" - "$timing_csv" "$DATA_PATH" "$PROMPT_START" "$MAX_PROMPTS" <<'PY'
import csv
import sys
from pathlib import Path

timing_csv, prompt_path = map(Path, sys.argv[1:3])
start, requested = map(int, sys.argv[3:5])
prompts = [line for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = max(0, len(prompts) - start) if requested <= 0 else min(requested, max(0, len(prompts) - start))
with timing_csv.open(newline="", encoding="utf-8") as f:
    completed = sum(
        row.get("status") in {"generated", "skipped_exists"}
        and Path(row.get("output_path", "")).is_file()
        for row in csv.DictReader(f)
    )
raise SystemExit(0 if expected > 0 and completed >= expected else 1)
PY
}

run_evaluation() {
  local output_folder="$1"
  local timing_csv="$output_folder/inference_times.csv"
  local eval_dir="$output_folder/eval"

  if loop_closure_eval_enabled; then
    if [ ! -f "$EVAL_LOOP_MANIFEST" ]; then
      echo "Loop-closure manifest not found: $EVAL_LOOP_MANIFEST" >&2
      return 3
    fi
    local loop_args=(
      --timing-csv "$timing_csv"
      --manifest "$EVAL_LOOP_MANIFEST"
      --output-dir "$eval_dir"
      --max-frames-per-segment "$EVAL_LOOP_MAX_FRAMES"
      --resize-width "$EVAL_LOOP_RESIZE_WIDTH"
      --device "$EVAL_LOOP_DEVICE"
      --lpips-batch-size "$EVAL_LOOP_LPIPS_BATCH_SIZE"
    )
    if [ -n "$EVAL_LOOP_DURATION_LABEL" ]; then
      loop_args+=(--duration-label "$EVAL_LOOP_DURATION_LABEL")
    fi
    if bool_enabled "$EVAL_LOOP_SKIP_LPIPS"; then
      loop_args+=(--skip-lpips)
    fi
    "$PYTHON_BIN" Wan21/scripts/evaluation/evaluate_loop_closure.py "${loop_args[@]}"
  fi

  if bool_enabled "$EVAL_STYLE_ENABLE"; then
    "$PYTHON_BIN" Wan21/scripts/evaluation/evaluate_vbench_style.py \
      --timing-csv "$timing_csv" \
      --output-dir "$eval_dir" \
      --max-frames "$EVAL_STYLE_MAX_FRAMES" \
      --resize-width "$EVAL_STYLE_RESIZE_WIDTH"
  fi

  if bool_enabled "$EVAL_OFFICIAL_VBENCH_ENABLE"; then
    if [ ! -f "$OFFICIAL_VBENCH_ROOT/evaluate.py" ]; then
      echo "Official VBench entrypoint not found: $OFFICIAL_VBENCH_ROOT/evaluate.py" >&2
      return 3
    fi
    if [ ! -x "$OFFICIAL_VBENCH_PYTHON" ]; then
      echo "Official VBench Python is not executable: $OFFICIAL_VBENCH_PYTHON" >&2
      return 3
    fi
    # shellcheck disable=SC2206
    local dimensions=($OFFICIAL_VBENCH_DIMENSIONS)
    local official_args=(
      --timing-csv "$timing_csv"
      --output-dir "$eval_dir"
      --vbench-root "$OFFICIAL_VBENCH_ROOT"
      --python-bin "$OFFICIAL_VBENCH_PYTHON"
      --master-port "$(pick_master_port)"
      --dimensions "${dimensions[@]}"
    )
    if [ -n "$OFFICIAL_VBENCH_CACHE_DIR" ]; then
      official_args+=(--vbench-cache-dir "$OFFICIAL_VBENCH_CACHE_DIR")
    fi
    if bool_enabled "$OFFICIAL_VBENCH_LOAD_LOCAL"; then
      official_args+=(--load-ckpt-from-local)
    fi
    "$PYTHON_BIN" Wan21/scripts/evaluation/evaluate_vbench_official.py "${official_args[@]}"
  fi
}

run_case() {
  local seed="$1"
  local case_name="$2"
  local output_folder="$RUN_ROOT/seed_${seed}/${case_name}"

  local sink_strategy="none"
  local sink_size=0
  local sink_interval=0
  local kv_bank_enable=0
  local retrieval_enable=0
  local retrieval_granularity=chunk
  local retrieval_metric=pose
  local compression_enable=0
  local compression_at_store=0
  local compression_pooled=0
  local compression_dynamic=0

  case "$case_name" in
    baseline) ;;
    fixed_sink)
      sink_strategy=fixed; sink_size="$SINK_SIZE" ;;
    periodic_sink)
      sink_strategy=periodic; sink_size="$SINK_SIZE"; sink_interval="$SINK_UPDATE_INTERVAL" ;;
    bank_random_sink|bank_uniform_sink|bank_pose_sink|bank_worldkv_fov_sink)
      sink_strategy="${case_name%_sink}"; sink_size="$SINK_SIZE"; sink_interval="$SINK_UPDATE_INTERVAL"; kv_bank_enable=1 ;;
    pose|pose_chunk)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_metric=pose; kv_bank_enable=1 ;;
    pose_latent_frame)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_granularity=latent_frame
      retrieval_metric=pose; kv_bank_enable=1 ;;
    hy_fov|hybrid)
      retrieval_enable=1; retrieval_metric="$case_name"; kv_bank_enable=1 ;;
    worldkv_fov|worldkv_fov_chunk)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_metric=worldkv_fov; kv_bank_enable=1 ;;
    worldkv_fov_latent_frame)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_granularity=latent_frame
      retrieval_metric=worldkv_fov; kv_bank_enable=1 ;;
    pose_compress_store)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_metric=pose; kv_bank_enable=1
      compression_enable=1; compression_at_store=1; compression_pooled=1 ;;
    worldkv_fov_compress_store)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_metric=worldkv_fov; kv_bank_enable=1
      compression_enable=1; compression_at_store=1; compression_pooled=1 ;;
    worldkv_fov_dynamic_compress_store)
      sink_strategy=fixed; sink_size="$SINK_SIZE"
      retrieval_enable=1; retrieval_metric=worldkv_fov; kv_bank_enable=1
      compression_enable=1; compression_at_store=1; compression_pooled=1; compression_dynamic=1 ;;
    *) echo "Unknown case: $case_name" >&2; return 2 ;;
  esac

  if bool_enabled "$SKIP_COMPLETED" && inference_complete "$output_folder"; then
    echo "[seed=$seed case=$case_name] generation already complete; evaluating existing output"
    run_evaluation "$output_folder"
    printf "%s\t%s\tskipped_existing\tok\t%s\n" "$seed" "$case_name" "$output_folder" >> "$SUMMARY"
    return 0
  fi

  local master_port
  master_port="$(pick_master_port)"
  echo "=== seed=$seed case=$case_name output=$output_folder ==="

  local env_args=(
    CONFIG_PATH="$CONFIG_PATH"
    CHECKPOINT_PATH="$CHECKPOINT_PATH"
    DATA_PATH="$DATA_PATH"
    TRAJECTORY="$TRAJECTORY"
    TRAJECTORY_PATH="$TRAJECTORY_PATH"
    OUTPUT_FOLDER="$output_folder"
    NUM_OUTPUT_FRAMES="$NUM_OUTPUT_FRAMES"
    MAX_PROMPTS="$MAX_PROMPTS"
    PROMPT_START="$PROMPT_START"
    SEED="$seed"
    SP_SIZE="$SP_SIZE"
    MASTER_PORT="$master_port"
    LOG_CACHE_STATE="$LOG_CACHE_STATE"
    LOG_CACHE_INTERVAL="$LOG_CACHE_INTERVAL"
    SINK_STRATEGY="$sink_strategy"
    SINK_SIZE="$sink_size"
    SINK_UPDATE_INTERVAL="$sink_interval"
    SINK_BANK_SEED="$SINK_BANK_SEED"
    KV_BANK_ENABLE="$kv_bank_enable"
    KV_BANK_DEVICE="$KV_BANK_DEVICE"
    KV_BANK_MAX_BLOCKS="$KV_BANK_MAX_BLOCKS"
    KV_BANK_LOG_INTERVAL="$KV_BANK_LOG_INTERVAL"
    KV_BANK_WARN_MEMORY_GB="$KV_BANK_WARN_MEMORY_GB"
    RETRIEVAL_ENABLE="$retrieval_enable"
    RETRIEVAL_GRANULARITY="$retrieval_granularity"
    RETRIEVAL_METRIC="$retrieval_metric"
    RETRIEVAL_FRAMES="$RETRIEVAL_FRAMES"
    RETRIEVAL_RECENT_FRAMES="$RETRIEVAL_RECENT_FRAMES"
    RETRIEVAL_FOV_SAMPLES="$RETRIEVAL_FOV_SAMPLES"
    RETRIEVAL_FOV_RADIUS="$RETRIEVAL_FOV_RADIUS"
    RETRIEVAL_FOV_H_DEG="$RETRIEVAL_FOV_H_DEG"
    RETRIEVAL_FOV_V_DEG="$RETRIEVAL_FOV_V_DEG"
    RETRIEVAL_HYBRID_FOV_WEIGHT="$RETRIEVAL_HYBRID_FOV_WEIGHT"
    RETRIEVAL_ROPE_CORRECTION="$RETRIEVAL_ROPE_CORRECTION"
    PROPE_REENCODE_MODE="$PROPE_REENCODE_MODE"
    KV_COMPRESSION_ENABLE="$compression_enable"
    KV_COMPRESSION_KEEP_RATIO="$KV_COMPRESSION_KEEP_RATIO"
    KV_COMPRESSION_AT_STORE="$compression_at_store"
    KV_COMPRESSION_POOLED="$compression_pooled"
    KV_COMPRESSION_DYNAMIC_ENABLE="$compression_dynamic"
    KV_COMPRESSION_DYNAMIC_MIN_KEEP="$KV_COMPRESSION_DYNAMIC_MIN_KEEP"
    KV_COMPRESSION_DYNAMIC_MAX_KEEP="$KV_COMPRESSION_DYNAMIC_MAX_KEEP"
    KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE="$KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE"
    KV_COMPRESSION_DYNAMIC_ROTATION_SCALE="$KV_COMPRESSION_DYNAMIC_ROTATION_SCALE"
    KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT="$KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT"
  )

  if bool_enabled "$DRY_RUN"; then
    printf "env "; printf "%q " "${env_args[@]}"
    printf "bash %q\n" "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"
    printf "%s\t%s\tdry_run\tskipped\t%s\n" "$seed" "$case_name" "$output_folder" >> "$SUMMARY"
    return 0
  fi

  set +e
  env "${env_args[@]}" bash "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"
  local inference_status=$?
  set -e
  if [ "$inference_status" -ne 0 ]; then
    printf "%s\t%s\tfailed:%s\tskipped\t%s\n" "$seed" "$case_name" "$inference_status" "$output_folder" >> "$SUMMARY"
    if bool_enabled "$CONTINUE_ON_ERROR"; then return 0; fi
    return "$inference_status"
  fi

  set +e
  run_evaluation "$output_folder"
  local eval_status=$?
  set -e
  if [ "$eval_status" -eq 0 ]; then
    printf "%s\t%s\tok\tok\t%s\n" "$seed" "$case_name" "$output_folder" >> "$SUMMARY"
  else
    printf "%s\t%s\tok\tfailed:%s\t%s\n" "$seed" "$case_name" "$eval_status" "$output_folder" >> "$SUMMARY"
    if ! bool_enabled "$CONTINUE_ON_ERROR"; then return "$eval_status"; fi
  fi
}

echo "Run root: $RUN_ROOT"
echo "Cases: ${CASE_LIST[*]}"
echo "Seeds: $SEEDS"
for seed in $SEEDS; do
  for case_name in "${CASE_LIST[@]}"; do
    run_case "$seed" "$case_name"
  done
done
echo "Summary: $SUMMARY"
