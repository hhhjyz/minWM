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
GPU_MIN_FREE_MEMORY_GB="${GPU_MIN_FREE_MEMORY_GB:-$GPU_MIN_MEMORY_GB}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RETRY_SECONDS="${RETRY_SECONDS:-300}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
SEEDS="${SEEDS:-0}"
# DURATIONS accepts either labels or explicit label:latent_frames entries, e.g.
#   DURATIONS="10s 20s"
#   DURATIONS="10s:40 20s:80 30s:120"
DURATIONS="${DURATIONS:-10s 15s 20s 30s}"
# CUDA_DEVICES uses the physical ids/UUIDs understood by CUDA_VISIBLE_DEVICES.
# All selected devices cooperate on each video through sequence parallelism.
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
CASES="${CASES:-baseline fixed_sink fixed_sink_only_rope_rebase tri_region_rope_rebase periodic_sink bank_random_sink bank_uniform_sink bank_pose_sink bank_worldkv_fov_sink pose worldkv_fov hy_fov hybrid pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store}"

if [ -z "$CUDA_DEVICES" ] || [[ "$CUDA_DEVICES" =~ [[:space:]] ]]; then
  echo "CUDA_DEVICES must be a non-empty comma-separated list without spaces, e.g. 0,1." >&2
  exit 2
fi

IFS=',' read -r -a CUDA_DEVICE_LIST <<< "$CUDA_DEVICES"
SELECTED_DEVICE_COUNT="${#CUDA_DEVICE_LIST[@]}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-$SELECTED_DEVICE_COUNT}"
SP_SIZE="${SP_SIZE:-$NUM_GPUS_PER_NODE}"

if ! [[ "$NUM_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || ! [[ "$SP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS_PER_NODE and SP_SIZE must be positive integers." >&2
  exit 2
fi
if [ "$NUM_GPUS_PER_NODE" -ne "$SELECTED_DEVICE_COUNT" ]; then
  echo "NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE does not match CUDA_DEVICES=$CUDA_DEVICES ($SELECTED_DEVICE_COUNT devices)." >&2
  exit 2
fi
if [ "$SP_SIZE" -ne "$NUM_GPUS_PER_NODE" ]; then
  echo "This inference queue requires SP_SIZE to equal NUM_GPUS_PER_NODE; got SP_SIZE=$SP_SIZE and NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE." >&2
  exit 2
fi

resolve_duration() {
  local spec="$1"
  case "$spec" in
    10s) printf '10s:40\n' ;;
    15s) printf '15s:60\n' ;;
    20s) printf '20s:80\n' ;;
    30s) printf '30s:120\n' ;;
    *:*)
      local label="${spec%%:*}"
      local frames="${spec##*:}"
      if [ -z "$label" ] || ! [[ "$frames" =~ ^[1-9][0-9]*$ ]]; then
        return 1
      fi
      printf '%s:%s\n' "$label" "$frames"
      ;;
    *) return 1 ;;
  esac
}

declare -a RESOLVED_DURATIONS=()
declare -A SEEN_DURATION_LABELS=()
for spec in $DURATIONS; do
  if ! resolved_spec="$(resolve_duration "$spec")"; then
    echo "Invalid duration '$spec'. Use 10s/15s/20s/30s or label:latent_frames." >&2
    exit 2
  fi
  resolved_label="${resolved_spec%%:*}"
  if [ -n "${SEEN_DURATION_LABELS[$resolved_label]:-}" ]; then
    echo "Duplicate duration label '$resolved_label' in DURATIONS='$DURATIONS'." >&2
    exit 2
  fi
  if [ ! -f "Wan21/prompts/demos_loop_closure/trajectories_${resolved_label}.txt" ]; then
    echo "Missing trajectory file for duration '$resolved_label'." >&2
    exit 2
  fi
  SEEN_DURATION_LABELS[$resolved_label]=1
  RESOLVED_DURATIONS+=("$resolved_spec")
done
if [ "${#RESOLVED_DURATIONS[@]}" -eq 0 ]; then
  echo "DURATIONS must select at least one duration." >&2
  exit 2
fi

export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

mkdir -p "$OUTPUT_ROOT"
read -r -a CASE_LIST <<< "$CASES"
CASE_COUNT="${#CASE_LIST[@]}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '[%s] another watchdog already holds %s\n' "$(date '+%F %T')" "$LOCK_FILE" >> "$QUEUE_LOG"
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$QUEUE_LOG"
}

gpu_memory_gb() {
  "$PYTHON_BIN" - <<'PY' 2>/dev/null || printf '0 0 0\n'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print("0 0 0")
else:
    memory = [torch.cuda.mem_get_info(index) for index in range(torch.cuda.device_count())]
    min_free = min(free for free, _ in memory) / (1024 ** 3)
    min_total = min(total for _, total in memory) / (1024 ** 3)
    print(min_total, min_free, torch.cuda.device_count())
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

log "watchdog started durations=${RESOLVED_DURATIONS[*]} cuda_devices=$CUDA_DEVICES cuda_device_order=$CUDA_DEVICE_ORDER sp_size=$SP_SIZE min_total_gpu_gb=$GPU_MIN_MEMORY_GB min_free_gpu_gb=$GPU_MIN_FREE_MEMORY_GB"

for spec in "${RESOLVED_DURATIONS[@]}"; do
  label="${spec%%:*}"
  frames="${spec##*:}"
  run_root="$OUTPUT_ROOT/string_loop_${label}_all_cases_seed0"
  mkdir -p "$run_root"

  while ! duration_complete "$run_root"; do
    memory_stats="$(gpu_memory_gb)"
    read -r total_gb free_gb visible_device_count <<< "$memory_stats"
    if [ "$visible_device_count" -ne "$SELECTED_DEVICE_COUNT" ]; then
      log "$label waiting_for_gpu visible_devices=$visible_device_count required_devices=$SELECTED_DEVICE_COUNT cuda_devices=$CUDA_DEVICES"
      sleep "$POLL_SECONDS"
      continue
    fi
    if ! awk -v actual="$total_gb" -v required="$GPU_MIN_MEMORY_GB" 'BEGIN { exit !(actual >= required) }'; then
      log "$label waiting_for_gpu min_total_gb=$total_gb required_total_gb=$GPU_MIN_MEMORY_GB"
      sleep "$POLL_SECONDS"
      continue
    fi
    if ! awk -v actual="$free_gb" -v required="$GPU_MIN_FREE_MEMORY_GB" 'BEGIN { exit !(actual >= required) }'; then
      log "$label waiting_for_free_gpu_memory min_free_gb=$free_gb required_free_gb=$GPU_MIN_FREE_MEMORY_GB"
      sleep "$POLL_SECONDS"
      continue
    fi

    log "$label resume cuda_devices=$CUDA_DEVICES sp_size=$SP_SIZE min_total_gpu_gb=$total_gb min_free_gpu_gb=$free_gb frames=$frames"
    env \
      PATH="$PATH" \
      PYTHONPATH="$PROJECT_ROOT/Wan21:$PROJECT_ROOT/shared:${PYTHONPATH:-}" \
      PYTHON_BIN="$PYTHON_BIN" \
      CUDA_DEVICE_ORDER="$CUDA_DEVICE_ORDER" \
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
      NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE" \
      SP_SIZE="$SP_SIZE" \
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
      EVAL_LOOP_CLOSURE_ENABLE=1 \
      EVAL_LOOP_MANIFEST=Wan21/prompts/demos_loop_closure/manifest.json \
      EVAL_LOOP_MAX_FRAMES=96 \
      EVAL_LOOP_RESIZE_WIDTH=256 \
      EVAL_STYLE_ENABLE=1 \
      EVAL_OFFICIAL_VBENCH_ENABLE=0 \
      SKIP_COMPLETED=1 \
      CONTINUE_ON_ERROR=1 \
      bash Wan21/scripts/inference/run_string_camera_experiments.sh >> "$run_root/run_all_cases.log" 2>&1
    runner_status=$?

    if duration_complete "$run_root"; then
      log "$label complete videos=$((MAX_PROMPTS * CASE_COUNT)) cases=$CASE_COUNT runner_status=$runner_status"
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
