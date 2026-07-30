#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.."; pwd)"
cd "$PROJECT_ROOT"

RUN_ROOT="${RUN_ROOT:-./outputs/loop10s_worldkv_compression_budget_ablation}"
DOC_TEMPLATE="$SCRIPT_DIR/docs/loop10s_worldkv_compression_budget_ablation.md"
CONFIG_PATH="${CONFIG_PATH:-Wan21/configs/causal_forcing_dmd_camera.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-../ckpts/Wan21/Action2V/dmd/model.pt}"
DATA_PATH="${DATA_PATH:-Wan21/prompts/demos_loop_closure/prompts.txt}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-Wan21/prompts/demos_loop_closure/trajectories_10s.txt}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-40}"
MAX_PROMPTS="${MAX_PROMPTS:-30}"
PROMPT_START="${PROMPT_START:-0}"
SEED="${SEED:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
EVAL_LOOP_CLOSURE_ENABLE="${EVAL_LOOP_CLOSURE_ENABLE:-1}"
EVAL_LOOP_SKIP_LPIPS="${EVAL_LOOP_SKIP_LPIPS:-0}"

bool_enabled() {
  case "$1" in
    1|true|True|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

inference_complete() {
  local output_folder="$1"
  local timing_csv="$output_folder/inference_times.csv"
  [ -f "$timing_csv" ] || return 1
  "$PYTHON_BIN" - "$timing_csv" "$DATA_PATH" "$PROMPT_START" "$MAX_PROMPTS" <<'PY'
import csv
import sys
from pathlib import Path

timing_csv, prompt_path = map(Path, sys.argv[1:3])
start, requested = map(int, sys.argv[3:5])
prompts = [line for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
expected = max(0, len(prompts) - start) if requested <= 0 else min(
    requested, max(0, len(prompts) - start)
)
with timing_csv.open(newline="", encoding="utf-8") as handle:
    completed = sum(
        row.get("status") in {"generated", "skipped_exists"}
        and Path(row.get("output_path", "")).is_file()
        for row in csv.DictReader(handle)
    )
raise SystemExit(0 if expected > 0 and completed >= expected else 1)
PY
}

run_evaluation() {
  local output_folder="$1"
  if ! bool_enabled "$EVAL_LOOP_CLOSURE_ENABLE"; then
    return 0
  fi
  local eval_args=(
    --timing-csv "$output_folder/inference_times.csv"
    --manifest Wan21/prompts/demos_loop_closure/manifest.json
    --duration-label 10s
    --output-dir "$output_folder/eval"
    --max-frames-per-segment 96
    --resize-width 256
  )
  if bool_enabled "$EVAL_LOOP_SKIP_LPIPS"; then
    eval_args+=(--skip-lpips)
  fi
  "$PYTHON_BIN" Wan21/scripts/evaluation/evaluate_loop_closure.py "${eval_args[@]}"
}

write_budget_summary() {
  local case_name="$1"
  local output_folder="$2"
  "$PYTHON_BIN" - "$case_name" "$output_folder" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

case_name, output_raw = sys.argv[1:3]
output = Path(output_raw)
log_path = output / "profile" / "inference.log"
event_path = output / "retrieval_events.jsonl"

keep_ratios = []
motion_scores = []
bank_total_gb = []
if log_path.is_file():
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for keep, motion in re.findall(
        r"keep_ratio=([^ ]+) motion_score=([^\\s]+)", text
    ):
        if keep != "none":
            keep_ratios.append(float(keep))
        if motion != "none":
            motion_scores.append(float(motion))
    bank_total_gb = [
        float(value)
        for value in re.findall(r"\\[kv-bank\\].*?total_gb=([0-9.eE+-]+)", text)
    ]

retrieved_tokens = []
if event_path.is_file():
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        tokens = int(event.get("retrieved_tokens_per_layer", 0))
        if tokens > 0:
            retrieved_tokens.append(tokens)

def describe(values):
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }

summary = {
    "case": case_name,
    "keep_ratio": describe(keep_ratios),
    "motion_score": describe(motion_scores),
    "retrieved_tokens_per_layer": describe(retrieved_tokens),
    "retrieved_frame_equivalents": describe(
        [tokens / 1560.0 for tokens in retrieved_tokens]
    ),
    "max_observed_kv_bank_gb": max(bank_total_gb) if bank_total_gb else None,
}
(output / "compression_budget_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(
    f"[{case_name}] budget summary: "
    f"mean_keep={summary['keep_ratio']['mean']} "
    f"mean_retrieval_eq={summary['retrieved_frame_equivalents']['mean']}"
)
PY
}

run_case() {
  local case_name="$1"
  local retrieval_frames="$2"
  local compression_enable="$3"
  local keep_ratio="$4"
  local dynamic_enable="${5:-0}"
  local dynamic_min_keep="${6:-0.2}"
  local dynamic_max_keep="${7:-0.5}"
  local dynamic_motion_weight="${8:-0.0}"
  local output_folder="$RUN_ROOT/$case_name"

  if bool_enabled "$SKIP_COMPLETED" && inference_complete "$output_folder"; then
    echo "[$case_name] inference already complete; evaluating existing videos"
    write_budget_summary "$case_name" "$output_folder"
    run_evaluation "$output_folder"
    return 0
  fi

  echo "[$case_name] retrieval_frames=$retrieval_frames compression=$compression_enable keep_ratio=$keep_ratio dynamic=$dynamic_enable motion_weight=$dynamic_motion_weight"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  CONFIG_PATH="$CONFIG_PATH" \
  CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  DATA_PATH="$DATA_PATH" \
  TRAJECTORY_PATH="$TRAJECTORY_PATH" \
  NUM_OUTPUT_FRAMES="$NUM_OUTPUT_FRAMES" \
  MAX_PROMPTS="$MAX_PROMPTS" \
  PROMPT_START="$PROMPT_START" \
  SEED="$SEED" \
  OUTPUT_FOLDER="$output_folder" \
  SINK_STRATEGY=fixed \
  SINK_SIZE=4 \
  SINK_UPDATE_INTERVAL=0 \
  TRI_REGION_ROPE_REBASE=0 \
  FIXED_SINK_ROPE_REBASE=0 \
  KV_BANK_ENABLE=1 \
  KV_BANK_DEVICE=cpu \
  KV_BANK_MAX_BLOCKS=10 \
  KV_BANK_LOG_INTERVAL=1 \
  KV_BANK_WARN_MEMORY_GB=128 \
  RETRIEVAL_ENABLE=1 \
  RETRIEVAL_GRANULARITY=chunk \
  RETRIEVAL_METRIC=worldkv_fov \
  RETRIEVAL_FRAMES="$retrieval_frames" \
  RETRIEVAL_RECENT_FRAMES=8 \
  RETRIEVAL_ROPE_CORRECTION=0 \
  PROPE_REENCODE_MODE=none \
  KV_COMPRESSION_ENABLE="$compression_enable" \
  KV_COMPRESSION_KEEP_RATIO="$keep_ratio" \
  KV_COMPRESSION_AT_STORE="$compression_enable" \
  KV_COMPRESSION_POOLED=1 \
  KV_COMPRESSION_DYNAMIC_ENABLE="$dynamic_enable" \
  KV_COMPRESSION_DYNAMIC_MIN_KEEP="$dynamic_min_keep" \
  KV_COMPRESSION_DYNAMIC_MAX_KEEP="$dynamic_max_keep" \
  KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE=1.0 \
  KV_COMPRESSION_DYNAMIC_ROTATION_SCALE=0.35 \
  KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT="$dynamic_motion_weight" \
  LOG_CACHE_STATE=0 \
  bash "$SCRIPT_DIR/run_profiled_smoke_causal_camera.sh"

  write_budget_summary "$case_name" "$output_folder"
  run_evaluation "$output_folder"
}

mkdir -p "$RUN_ROOT"
cp "$DOC_TEMPLATE" "$RUN_ROOT/README.md"

# A vs B: compression damage at identical raw history coverage.
run_case A_retr8_no_compression 8 0 1.0
run_case B_retr8_compression_r050 8 1 0.5

# A vs C: approximately fixed retrieval-token budget with 1.5x raw coverage.
run_case C_retr12_compression_r050 12 1 0.5

# A vs D: exact fixed retrieval-token budget with 2x raw coverage.
run_case D_retr16_compression_r0333 16 1 0.3333333333333333

# D vs E/F: allocate a similar average token budget according to camera speed.
# Negative motion_weight means faster motion keeps more tokens because the
# implementation computes keep = base - motion_weight * motion_score.
run_case E_retr16_dynamic_fast_keep_more 16 1 0.25 1 0.2 0.5 -0.25

# Positive motion_weight is the legacy hypothesis: faster motion is compressed
# more aggressively. The higher base approximately matches E/D's mean ratio on
# the 10s loop-closure motion-score distribution.
run_case F_retr16_dynamic_fast_compress_more 16 1 0.4166666666666667 1 0.2 0.5 0.25

"$PYTHON_BIN" - "$RUN_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/compression_budget_summary.json")):
    item = json.loads(path.read_text(encoding="utf-8"))
    rows.append({
        "case": item["case"],
        "mean_keep_ratio": item["keep_ratio"]["mean"],
        "mean_motion_score": item["motion_score"]["mean"],
        "mean_retrieved_tokens_per_layer": item["retrieved_tokens_per_layer"]["mean"],
        "mean_retrieved_frame_equivalents": item["retrieved_frame_equivalents"]["mean"],
        "max_observed_kv_bank_gb": item["max_observed_kv_bank_gb"],
    })
if rows:
    with (root / "compression_budget_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
PY

echo "All experiments complete: $RUN_ROOT"
