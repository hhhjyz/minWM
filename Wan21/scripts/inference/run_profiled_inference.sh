#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <inference command...>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

export OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs/profiled_$(date +%Y%m%d_%H%M%S)}"
PROFILE_DIR="${OUTPUT_FOLDER}/profile"
mkdir -p "$PROFILE_DIR"
rm -f \
  "$PROFILE_DIR/gpu_samples.csv" \
  "$PROFILE_DIR/cache_state.csv" \
  "$PROFILE_DIR/gpu_memory.png" \
  "$PROFILE_DIR/gpu_utilization.png" \
  "$PROFILE_DIR/gpu_power.png" \
  "$PROFILE_DIR/gpu_temperature.png" \
  "$PROFILE_DIR/torch_cache_memory.png" \
  "$PROFILE_DIR/kv_cache_tokens.png"

LOG_FILE="${PROFILE_DIR}/inference.log"
RUN_META="${PROFILE_DIR}/run_meta.json"

START_EPOCH="$(date +%s)"
START_EPOCH_FLOAT="$(date +%s.%N)"
set +e
"$@" 2>&1 | tee "$LOG_FILE"
CMD_STATUS="${PIPESTATUS[0]}"
set -e
END_EPOCH="$(date +%s)"
END_EPOCH_FLOAT="$(date +%s.%N)"

if command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif [ -x /home/hhhjyz/miniconda3/bin/conda ]; then
  PYTHON_CMD=(/home/hhhjyz/miniconda3/bin/conda run -n ling python)
else
  echo "Cannot find python for profile analysis." >&2
  exit "$CMD_STATUS"
fi

PROFILE_START_EPOCH="$START_EPOCH" \
PROFILE_END_EPOCH="$END_EPOCH" \
PROFILE_START_EPOCH_FLOAT="$START_EPOCH_FLOAT" \
PROFILE_END_EPOCH_FLOAT="$END_EPOCH_FLOAT" \
PROFILE_CMD_STATUS="$CMD_STATUS" \
"${PYTHON_CMD[@]}" -c '
import json
import os
import sys

path = sys.argv[1]
start = float(os.environ["PROFILE_START_EPOCH_FLOAT"])
end = float(os.environ["PROFILE_END_EPOCH_FLOAT"])
keys = [
    "CONFIG_PATH",
    "CHECKPOINT_PATH",
    "DATA_PATH",
    "TRAJECTORY_PATH",
    "TRAJECTORY_POSE_PATH",
    "OUTPUT_FOLDER",
    "MAX_PROMPTS",
    "PROMPT_START",
    "NUM_OUTPUT_FRAMES",
    "SP_SIZE",
    "LOG_CACHE_STATE",
    "LOG_CACHE_INTERVAL",
    "SINK_STRATEGY",
    "SINK_SIZE",
    "SINK_UPDATE_INTERVAL",
    "KV_BANK_ENABLE",
    "KV_BANK_DEVICE",
    "KV_BANK_MAX_BLOCKS",
    "KV_BANK_LOG_INTERVAL",
    "KV_BANK_WARN_MEMORY_GB",
    "RETRIEVAL_ENABLE",
    "RETRIEVAL_METRIC",
    "RETRIEVAL_FRAMES",
    "RETRIEVAL_RECENT_FRAMES",
    "RETRIEVAL_FOV_SAMPLES",
    "RETRIEVAL_FOV_RADIUS",
    "RETRIEVAL_FOV_H_DEG",
    "RETRIEVAL_FOV_V_DEG",
    "RETRIEVAL_HYBRID_FOV_WEIGHT",
    "RETRIEVAL_ROPE_CORRECTION",
    "KV_COMPRESSION_ENABLE",
    "KV_COMPRESSION_KEEP_RATIO",
    "KV_COMPRESSION_ANCHOR_ROTATE",
    "KV_COMPRESSION_AT_STORE",
    "KV_COMPRESSION_POOLED",
    "CUDA_VISIBLE_DEVICES",
]
meta = {
    "command": sys.argv[2:] if len(sys.argv) > 2 else [],
    "exit_status": int(os.environ["PROFILE_CMD_STATUS"]),
    "start_epoch": int(os.environ["PROFILE_START_EPOCH"]),
    "end_epoch": int(os.environ["PROFILE_END_EPOCH"]),
    "wall_time_seconds": max(0.0, end - start),
    "env": {key: os.environ.get(key) for key in keys if os.environ.get(key) is not None},
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)
' "$RUN_META" "$@"

"${PYTHON_CMD[@]}" "$SCRIPT_DIR/plot_inference_profile.py" \
  --times-csv "$OUTPUT_FOLDER/inference_times.csv" \
  --run-meta "$RUN_META" \
  --log-file "$LOG_FILE" \
  --output-dir "$PROFILE_DIR"

echo "Profile artifacts saved to: $PROFILE_DIR"
exit "$CMD_STATUS"
