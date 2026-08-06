#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jiangyize/research/minWM}"
DATASET_ROOT="${DATASET_ROOT:-/home/jiangyize/research/datasets/MBench-Data/MBench-A}"
DA3_PYTHON="${DA3_PYTHON:-/home/jiangyize/software/miniconda3/envs/da3/bin/python}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
MODEL_GLOB="${MODEL_GLOB:-minwm_sink_rebase_retrieval_*_seed0}"
DA3_MODEL="${DA3_MODEL:-depth-anything/DA3-LARGE-1.1}"
PROCESS_RES="${PROCESS_RES:-504}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/outputs/mbencha_da3}"
HF_HOME="${HF_HOME:-${RUN_DIR}/hf_cache}"
OVERWRITE_INVALID="${OVERWRITE_INVALID:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-15}"

# The cluster-wide hf-mirror does not expose the metadata headers required by
# the current huggingface_hub.  Use the official endpoint and the more robust
# plain HTTP downloader for this checkpoint.
unset HF_ENDPOINT
export HF_HOME HF_HUB_DISABLE_XET=1

SCRIPT="${PROJECT_ROOT}/Wan21/scripts/evaluation/prepare_mbencha_da3.py"
SUMMARY="${PROJECT_ROOT}/Wan21/scripts/evaluation/summarize_mbencha_da3.py"
MONITOR="${PROJECT_ROOT}/Wan21/scripts/evaluation/monitor_mbencha_da3_progress.py"
RUN_TOKEN="$(date +%Y%m%d-%H%M%S)-$$"
mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/progress"

if [[ ! -x "${DA3_PYTHON}" ]]; then
  echo "DA3 Python 不存在或不可执行: ${DA3_PYTHON}" >&2
  exit 1
fi

IFS=',' read -r -a DEVICES <<< "${CUDA_DEVICES}"
WORLD_SIZE="${#DEVICES[@]}"
if [[ "${WORLD_SIZE}" -lt 1 ]]; then
  echo "CUDA_DEVICES 不能为空" >&2
  exit 1
fi

LOCK_DIR="${RUN_DIR}/run.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "已有 DA3 launcher 在运行，或遗留锁存在: ${LOCK_DIR}" >&2
  echo "确认没有进程后可手动删除该空目录。" >&2
  exit 1
fi
monitor_pid=""
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

pids=()
progress_files=()
for rank in "${!DEVICES[@]}"; do
  device="${DEVICES[$rank]}"
  command=(
    "${DA3_PYTHON}" "${SCRIPT}"
    --dataset-root "${DATASET_ROOT}"
    --model-glob "${MODEL_GLOB}"
    --model "${DA3_MODEL}"
    --process-res "${PROCESS_RES}"
    --rank "${rank}"
    --world-size "${WORLD_SIZE}"
    --log-file "${RUN_DIR}/events_rank${rank}.jsonl"
    --progress-file "${RUN_DIR}/progress/${RUN_TOKEN}_rank${rank}.json"
  )
  if [[ "${OVERWRITE_INVALID}" == "1" ]]; then
    command+=(--overwrite-invalid)
  fi
  echo "启动 rank=${rank}/${WORLD_SIZE}, physical_cuda=${device}"
  CUDA_VISIBLE_DEVICES="${device}" "${command[@]}" \
    >"${RUN_DIR}/logs/rank${rank}.log" 2>&1 &
  pids+=("$!")
  progress_files+=("${RUN_DIR}/progress/${RUN_TOKEN}_rank${rank}.json")
done

"${DA3_PYTHON}" "${MONITOR}" \
  --interval "${PROGRESS_INTERVAL}" \
  "${progress_files[@]}" &
monitor_pid=$!

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ -n "${monitor_pid}" ]] && kill -0 "${monitor_pid}" 2>/dev/null; then
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
fi
monitor_pid=""
"${DA3_PYTHON}" "${MONITOR}" --once "${progress_files[@]}"

"${DA3_PYTHON}" "${SUMMARY}" \
  --dataset-root "${DATASET_ROOT}" \
  --model-glob "${MODEL_GLOB}" \
  --output "${RUN_DIR}/DA3_ARTIFACTS_PROGRESS.md"

if [[ "${failed}" != "0" ]]; then
  echo "至少一个 DA3 worker 失败；可直接重跑，已通过校验的样本会跳过。" >&2
  exit 1
fi
