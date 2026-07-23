#!/bin/bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/../ckpts/Wan21/Action2V/dmd/model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
QUEUE_LOG="${QUEUE_LOG:-$OUTPUT_ROOT/string_loop_all_durations_watchdog.log}"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_ROOT/string_loop_all_durations_watchdog.lock}"
GPU_MIN_MEMORY_GB="${GPU_MIN_MEMORY_GB:-70}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-300}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
SEEDS="${SEEDS:-0}"
DURATIONS="${DURATIONS:-10s:40 15s:60 20s:80 30s:120}"
CASES="${CASES:-baseline fixed_sink periodic_sink bank_random_sink bank_uniform_sink bank_pose_sink bank_worldkv_fov_sink pose worldkv_fov hy_fov hybrid pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store}"

mkdir -p "$OUTPUT_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '[%s] another watchdog already holds %s\n' "$(date '+%F %T')" "$LOCK_FILE" >> "$QUEUE_LOG"
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$QUEUE_LOG"
}

gpu_total_gb() {
  "$PYTHON_BIN" - <<'PY' 2>/dev/null || printf '0\n'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print(0)
else:
    print(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3))
PY
}

duration_complete() {
  local run_root="$1"
  local case_name count
  for case_name in $CASES; do
    count="$(find "$run_root/seed_0/$case_name" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)"
    if [ "$count" -lt "$MAX_PROMPTS" ]; then
      return 1
    fi
  done
  return 0
}

log "watchdog started durations=$DURATIONS min_gpu_gb=$GPU_MIN_MEMORY_GB"

for spec in $DURATIONS; do
  label="${spec%%:*}"
  frames="${spec##*:}"
  run_root="$OUTPUT_ROOT/string_loop_${label}_all_cases_seed0"
  mkdir -p "$run_root"

  while ! duration_complete "$run_root"; do
    total_gb="$(gpu_total_gb)"
    if ! awk -v actual="$total_gb" -v required="$GPU_MIN_MEMORY_GB" 'BEGIN { exit !(actual >= required) }'; then
      log "$label waiting_for_gpu visible_gb=$total_gb required_gb=$GPU_MIN_MEMORY_GB"
      sleep "$POLL_SECONDS"
      continue
    fi

    log "$label resume visible_gpu_gb=$total_gb frames=$frames"
    env \
      PATH="$PATH" \
      PYTHONPATH="$PROJECT_ROOT/Wan21:$PROJECT_ROOT/shared:${PYTHONPATH:-}" \
      PYTHON_BIN="$PYTHON_BIN" \
      CUDA_VISIBLE_DEVICES=0 \
      RUN_ROOT="$run_root" \
      CONFIG_PATH=Wan21/configs/causal_forcing_dmd_camera.yaml \
      CHECKPOINT_PATH="$CHECKPOINT_PATH" \
      DATA_PATH=Wan21/prompts/demos_loop_closure/prompts.txt \
      TRAJECTORY_PATH="Wan21/prompts/demos_loop_closure/trajectories_${label}.txt" \
      NUM_OUTPUT_FRAMES="$frames" \
      MAX_PROMPTS="$MAX_PROMPTS" \
      PROMPT_START=0 \
      SEEDS="$SEEDS" \
      CASES="$CASES" \
      SINK_SIZE=4 \
      SINK_UPDATE_INTERVAL=4 \
      SINK_BANK_SEED=0 \
      KV_BANK_DEVICE=cpu \
      KV_BANK_MAX_BLOCKS=10 \
      KV_BANK_WARN_MEMORY_GB=128 \
      KV_BANK_LOG_INTERVAL=1 \
      RETRIEVAL_FRAMES=8 \
      RETRIEVAL_RECENT_FRAMES=8 \
      RETRIEVAL_FOV_SAMPLES=8192 \
      RETRIEVAL_FOV_RADIUS=8.0 \
      RETRIEVAL_FOV_H_DEG=60.0 \
      RETRIEVAL_FOV_V_DEG=35.0 \
      RETRIEVAL_HYBRID_FOV_WEIGHT=0.5 \
      RETRIEVAL_ROPE_CORRECTION=0 \
      PROPE_REENCODE_MODE=none \
      KV_COMPRESSION_KEEP_RATIO=0.5 \
      LOG_CACHE_STATE=0 \
      EVAL_STYLE_ENABLE=1 \
      EVAL_OFFICIAL_VBENCH_ENABLE=0 \
      SKIP_COMPLETED=1 \
      CONTINUE_ON_ERROR=1 \
      bash Wan21/scripts/inference/run_string_camera_experiments.sh >> "$run_root/run_all_cases.log" 2>&1
    runner_status=$?

    if duration_complete "$run_root"; then
      log "$label complete videos=$((MAX_PROMPTS * 14)) runner_status=$runner_status"
      break
    fi

    log "$label runner_finished_but_incomplete status=$runner_status; retry_in=${RETRY_SECONDS}s"
    sleep "$RETRY_SECONDS"
  done

  if duration_complete "$run_root"; then
    log "$label verified_complete"
  fi
done

log "all durations verified complete"
