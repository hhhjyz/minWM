#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINWM_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$MINWM_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
MBENCHA_ROOT="${MBENCHA_ROOT:?Set MBENCHA_ROOT to the local MBench-A dataset root}"
LENGTH="${LENGTH:-25s}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-400}"
CONDITIONS="${CONDITIONS:-left_then_right,right_then_left,forward_then_backward,left_360,right_360}"
SUBSETS="${SUBSETS:-environment,human,object,causal}"
ADAPTER_ROOT="${ADAPTER_ROOT:-./outputs/mbencha_adapter/${LENGTH}}"
RUN_ROOT="${RUN_ROOT:-./outputs/mbencha_${LENGTH}}"
MODEL_PREFIX="${MODEL_PREFIX:-minwm}"
PACKAGE_AFTER_RUN="${PACKAGE_AFTER_RUN:-1}"
LINK_MODE="${LINK_MODE:-symlink}"
LIMIT="${LIMIT:-}"
CASES="${CASES:-baseline fixed_sink pose pose_latent_frame worldkv_fov worldkv_fov_latent_frame hy_fov hybrid pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store}"

prepare_args=(
  prepare
  --dataset-root "$MBENCHA_ROOT"
  --work-dir "$ADAPTER_ROOT"
  --length "$LENGTH"
  --conditions "$CONDITIONS"
  --subsets "$SUBSETS"
  --num-output-frames "$NUM_OUTPUT_FRAMES"
)
if [ -n "$LIMIT" ]; then
  prepare_args+=(--limit "$LIMIT")
fi
"$PYTHON_BIN" Wan21/scripts/evaluation/mbencha_adapter.py "${prepare_args[@]}"

export DATA_PATH="$ADAPTER_ROOT/prompts.txt"
export TRAJECTORY_PATH="$ADAPTER_ROOT/trajectories.txt"
export RUN_ROOT
export NUM_OUTPUT_FRAMES
export MAX_PROMPTS=0
export PROMPT_START=0
export CASES
export EVAL_STYLE_ENABLE="${EVAL_STYLE_ENABLE:-0}"
export EVAL_LOOP_CLOSURE_ENABLE=0
export EVAL_OFFICIAL_VBENCH_ENABLE=0

bash "$SCRIPT_DIR/run_string_camera_experiments.sh"

if [ "$PACKAGE_AFTER_RUN" = "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  "$PYTHON_BIN" Wan21/scripts/evaluation/mbencha_adapter.py package \
    --dataset-root "$MBENCHA_ROOT" \
    --manifest "$ADAPTER_ROOT/manifest.jsonl" \
    --run-root "$RUN_ROOT" \
    --model-prefix "$MODEL_PREFIX" \
    --link-mode "$LINK_MODE"
fi
