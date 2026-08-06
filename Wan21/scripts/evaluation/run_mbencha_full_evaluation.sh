#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# =============================================================================
# 用户配置区
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-/home/jiangyize/research/minWM}"
MBENCH_ROOT="${MBENCH_ROOT:-/home/jiangyize/research/MBench}"
DATASET_ROOT="${DATASET_ROOT:-/home/jiangyize/research/datasets/MBench-Data/MBench-A}"
MBENCH_PYTHON="${MBENCH_PYTHON:-/home/jiangyize/software/miniconda3/envs/minwm-fa/bin/python}"
DA3_PYTHON="${DA3_PYTHON:-/home/jiangyize/software/miniconda3/envs/da3/bin/python}"
MODEL_GLOB="${MODEL_GLOB:-minwm_sink_rebase_retrieval_*_seed0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/mbencha_full_evaluation}"

# all: 原流水线；da3: 只准备 DA3 并测 4 项依赖指标；
# non_da3: 兼容模式，测其余 8 项；vlm/non_vlm: 将这 8 项拆成 2+6。
RUN_GROUP="${RUN_GROUP:-all}"

# 物理 GPU。每个评测子进程只看到一张卡，因此内部仍使用逻辑 cuda:0。
DA3_CUDA_DEVICES="${DA3_CUDA_DEVICES:-0,1}"
EVAL_CUDA_DEVICE="${EVAL_CUDA_DEVICE:-0}"
# non_da3 模式支持逗号分隔的多卡任务级并行；每张卡同一时刻只跑一项指标。
EVAL_CUDA_DEVICES="${EVAL_CUDA_DEVICES:-${EVAL_CUDA_DEVICE}}"

# 1: 先补齐全部 DA3 artifacts；0: 只校验，缺失时停止。
RUN_DA3="${RUN_DA3:-1}"

# VLM 接口。推荐把真实值写进 chmod 600 的 VLM_CONFIG，而不是提交到 Git。
VLM_CONFIG="${VLM_CONFIG:-${PROJECT_ROOT}/outputs/mbencha_full_evaluation/vlm.env}"
MBENCH_VLM_BASE_URL="${MBENCH_VLM_BASE_URL:-}"
MBENCH_VLM_API_KEY="${MBENCH_VLM_API_KEY:-}"
MBENCH_VLM_MODEL="${MBENCH_VLM_MODEL:-}"
MBENCH_VLM_ENDPOINT_PATH="${MBENCH_VLM_ENDPOINT_PATH:-/chat/completions}"
MBENCH_VLM_API_STYLE="${MBENCH_VLM_API_STYLE:-chat_completions}"
VLM_WORKERS="${VLM_WORKERS:-1}"
VLM_N_FRAMES="${VLM_N_FRAMES:-8}"
VLM_MAX_RETRIES="${VLM_MAX_RETRIES:-10}"
VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-512}"
CHECK_VLM_ENDPOINT="${CHECK_VLM_ENDPOINT:-1}"

if [[ -f "${VLM_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${VLM_CONFIG}"
fi
export MBENCH_VLM_BASE_URL MBENCH_VLM_API_KEY MBENCH_VLM_MODEL
export MBENCH_VLM_ENDPOINT_PATH MBENCH_VLM_API_STYLE MBENCH_VLM_MAX_TOKENS

DA3_SCRIPT="${PROJECT_ROOT}/Wan21/scripts/evaluation/prepare_mbencha_da3.py"
DA3_RUNNER="${PROJECT_ROOT}/Wan21/scripts/evaluation/run_mbencha_da3.sh"
SUMMARY_SCRIPT="${PROJECT_ROOT}/Wan21/scripts/evaluation/summarize_mbencha_da3.py"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/metrics"
case "${RUN_GROUP}" in
  all|da3|non_da3|vlm|non_vlm) ;;
  *) echo "RUN_GROUP 必须是 all、da3、non_da3、vlm 或 non_vlm，当前为: ${RUN_GROUP}" >&2; exit 1 ;;
esac

NEEDS_VLM=0
[[ "${RUN_GROUP}" == "all" || "${RUN_GROUP}" == "non_da3" || "${RUN_GROUP}" == "vlm" ]] && NEEDS_VLM=1
NEEDS_DA3=0
[[ "${RUN_GROUP}" == "all" || "${RUN_GROUP}" == "da3" ]] && NEEDS_DA3=1
NEEDS_LOCAL_GPU=0
[[ "${RUN_GROUP}" != "vlm" ]] && NEEDS_LOCAL_GPU=1
LOCK_DIR="${OUTPUT_ROOT}/run.${RUN_GROUP}.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "已有全量 MBench-A runner 在运行，或遗留锁存在: ${LOCK_DIR}" >&2
  echo "确认没有任务后，可手动删除该空目录。" >&2
  exit 1
fi
DA3_PID=""
cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "${DA3_PID}" ]] && kill -0 "${DA3_PID}" 2>/dev/null; then
    echo "停止后台 DA3 stage (pid=${DA3_PID})" >&2
    kill "${DA3_PID}" 2>/dev/null || true
    wait "${DA3_PID}" 2>/dev/null || true
  fi
  rmdir "${LOCK_DIR}" 2>/dev/null || true
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

require_file() {
  if [[ ! -e "$1" ]]; then
    echo "缺少必要路径: $1" >&2
    exit 1
  fi
}
require_file "${MBENCH_PYTHON}"
if [[ "${NEEDS_DA3}" == "1" ]]; then
  require_file "${DA3_PYTHON}"
fi
require_file "${DATASET_ROOT}/dataset.yaml"
require_file "${MBENCH_ROOT}/mbench/cli.py"

IFS=',' read -r -a EVAL_DEVICES <<<"${EVAL_CUDA_DEVICES}"
if [[ "${#EVAL_DEVICES[@]}" -lt 1 || -z "${EVAL_DEVICES[0]}" ]]; then
  echo "EVAL_CUDA_DEVICES 不能为空" >&2
  exit 1
fi
# 单卡阶段（DA3 依赖指标）及预检默认使用列表中的第一张卡。
EVAL_CUDA_DEVICE="${EVAL_DEVICES[0]}"

if [[ "${NEEDS_LOCAL_GPU}" == "1" ]]; then
  echo "预检 MBench Python 依赖和评测 GPU"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" "${MBENCH_PYTHON}" -c '
import torch, torchvision, cv2, lpips, open_clip, insightface, numpy
from transformers import Sam2Model, Sam2Processor, AutoModel, AutoImageProcessor
if not torch.cuda.is_available():
    raise RuntimeError("MBench evaluation GPU is not available")
print(f"MBench preflight OK: torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")
' >"${OUTPUT_ROOT}/logs/python_preflight.log" 2>&1 || {
    cat "${OUTPUT_ROOT}/logs/python_preflight.log" >&2
    exit 1
  }
else
  echo "预检 VLM 模式 CPU 依赖（不要求 CUDA）"
  "${MBENCH_PYTHON}" -c 'import cv2, numpy; print("MBench VLM CPU preflight OK")' \
    >"${OUTPUT_ROOT}/logs/python_preflight_vlm.log" 2>&1 || {
      cat "${OUTPUT_ROOT}/logs/python_preflight_vlm.log" >&2
      exit 1
    }
fi

if [[ "${NEEDS_VLM}" == "1" && ( -z "${MBENCH_VLM_BASE_URL}" || -z "${MBENCH_VLM_API_KEY}" || -z "${MBENCH_VLM_MODEL}" ) ]]; then
  cat >&2 <<EOF
VLM 配置不完整。请编辑：
  ${VLM_CONFIG}
至少设置 MBENCH_VLM_BASE_URL、MBENCH_VLM_API_KEY、MBENCH_VLM_MODEL。
可从 ${PROJECT_ROOT}/Wan21/scripts/evaluation/mbencha_vlm.env.example 复制。
EOF
  exit 1
fi

if [[ "${NEEDS_VLM}" == "1" && "${CHECK_VLM_ENDPOINT}" == "1" ]]; then
  models_url="${MBENCH_VLM_BASE_URL%/}/models"
  echo "检查 VLM endpoint: ${models_url}（不会打印 API key）"
  if ! printf 'Authorization: Bearer %s\n' "${MBENCH_VLM_API_KEY}" | \
    curl --fail --silent --show-error --max-time 30 \
      --header @- "${models_url}" >/dev/null; then
    echo "VLM /models 检查失败。若服务不实现 /models，可设置 CHECK_VLM_ENDPOINT=0。" >&2
    exit 1
  fi
fi

echo "阶段 1/5：解析 14 个 minWM 模型 ID"
mapfile -t MODEL_IDS < <(
  find "${DATASET_ROOT}/models" -mindepth 1 -maxdepth 1 -type d \
    -name "${MODEL_GLOB}" -printf '%f\n' | sort
)
if [[ "${#MODEL_IDS[@]}" -ne 14 ]]; then
  echo "预期 14 个模型，实际发现 ${#MODEL_IDS[@]} 个。" >&2
  printf '  %s\n' "${MODEL_IDS[@]}" >&2
  exit 1
fi
MODELS_CSV="$(IFS=,; echo "${MODEL_IDS[*]}")"
printf '%s\n' "${MODEL_IDS[@]}" >"${OUTPUT_ROOT}/model_ids.txt"

# subset|metric|type；type=local 使用本地计算，type=vlm 使用远程接口。
# 两项 VLM 指标只调用远程接口；其余 6 项为本地指标。
VLM_JOBS=(
  "causal|mbencha.causal.state_progress|vlm"
  "causal|mbencha.causal.progress_correctness|vlm"
)
NON_VLM_JOBS=(
  "environment|mbencha.environment.rendering_lighting|local"
  "human|mbencha.entity.human_identity_consistency|local"
  "human|mbencha.entity.human_appearance_consistency|local"
  "object|mbencha.entity.object_texture_consistency|local"
  "environment|mbencha.environment.rendering_style|local"
  "causal|mbencha.causal.prompt_interaction|local"
)
DA3_DEPENDENT_JOBS=(
  "object|mbencha.entity.object_geometry_consistency|local"
  "environment|mbencha.environment.spatial_epipolar|local"
  "environment|mbencha.environment.spatial_reprojection|local"
  "causal|mbencha.causal.camera_interaction|local"
)
NON_DA3_JOBS=("${VLM_JOBS[@]}" "${NON_VLM_JOBS[@]}")
JOBS=("${NON_DA3_JOBS[@]}" "${DA3_DEPENDENT_JOBS[@]}")

run_jobs() {
  stage_name="$1"
  shift
  echo "运行指标阶段：${stage_name}（共 $# 项）"
  for spec in "$@"; do
    IFS='|' read -r subset metric kind <<<"${spec}"
    short_name="${metric#mbencha.}"
    metric_dir="${OUTPUT_ROOT}/metrics/${short_name}"
    marker="${metric_dir}/.complete"
    mkdir -p "${metric_dir}"
    if [[ -f "${marker}" ]]; then
      echo "[skip] ${metric} 已完成"
      continue
    fi

    echo "[validate] ${metric} subset=${subset}"
    (
      cd "${MBENCH_ROOT}"
      "${MBENCH_PYTHON}" -m mbench.cli validate "${DATASET_ROOT}" \
        --metrics "${metric}" \
        --models "${MODELS_CSV}" \
        --subsets "${subset}"
    ) >"${metric_dir}/validate.log" 2>&1

    echo "[eval] ${metric}"
    command=(
      "${MBENCH_PYTHON}" -m mbench.cli eval "${DATASET_ROOT}"
      --metrics "${metric}"
      --models "${MODELS_CSV}"
      --subsets "${subset}"
      --output "${metric_dir}"
      --no-progress-bar
      --progress-every 25
      --log-file "${metric_dir}/eval.log"
    )
    if [[ "${kind}" == "vlm" ]]; then
      command+=(
        --vlm-judge openai-compatible
        --vlm-api-style "${MBENCH_VLM_API_STYLE}"
        --workers "${VLM_WORKERS}"
        --vlm-max-retries "${VLM_MAX_RETRIES}"
        --vlm-n-frames "${VLM_N_FRAMES}"
        --vlm-max-tokens "${VLM_MAX_TOKENS}"
      )
    else
      command+=(--workers 1)
    fi

    (
      cd "${MBENCH_ROOT}"
      CUDA_VISIBLE_DEVICES="${EVAL_CUDA_DEVICE}" "${command[@]}"
    ) 2>&1 | tee "${metric_dir}/console.log"
    touch "${marker}"
  done
}

run_jobs_parallel() {
  stage_name="$1"
  shift
  specs=("$@")
  num_devices="${#EVAL_DEVICES[@]}"
  echo "并行指标阶段：${stage_name}（${#specs[@]} 项，${num_devices} 个 GPU worker）"
  worker_pids=()
  for rank in "${!EVAL_DEVICES[@]}"; do
    device="${EVAL_DEVICES[$rank]}"
    (
      worker_specs=()
      for ((index=rank; index<${#specs[@]}; index+=num_devices)); do
        worker_specs+=("${specs[$index]}")
      done
      if [[ "${#worker_specs[@]}" -gt 0 ]]; then
        EVAL_CUDA_DEVICE="${device}"
        run_jobs "${stage_name}/worker${rank}/gpu${device}" "${worker_specs[@]}"
      fi
    ) &
    worker_pids+=("$!")
  done

  failed=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" != "0" ]]; then
    echo "至少一个 non_da3 GPU worker 失败；已完成指标保留，可直接重跑。" >&2
    return 1
  fi
}

run_da3_stage() {
  if [[ "${RUN_DA3}" == "1" ]]; then
    PROJECT_ROOT="${PROJECT_ROOT}" \
    DATASET_ROOT="${DATASET_ROOT}" \
    DA3_PYTHON="${DA3_PYTHON}" \
    CUDA_DEVICES="${DA3_CUDA_DEVICES}" \
    MODEL_GLOB="${MODEL_GLOB}" \
    PROCESS_RES="${PROCESS_RES:-504}" \
      bash "${DA3_RUNNER}" 2>&1 | tee "${OUTPUT_ROOT}/logs/da3_stage.log"
  fi

  if ! "${DA3_PYTHON}" "${DA3_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --model-glob "${MODEL_GLOB}" \
    --verify-only \
    >"${OUTPUT_ROOT}/logs/da3_verify.log" 2>&1; then
    "${DA3_PYTHON}" "${SUMMARY_SCRIPT}" \
      --dataset-root "${DATASET_ROOT}" \
      --model-glob "${MODEL_GLOB}" \
      --output "${OUTPUT_ROOT}/DA3_ARTIFACTS_PROGRESS.md"
    echo "DA3 尚未齐全，4 项 DA3 依赖指标不会启动。" >&2
    exit 2
  fi
  "${DA3_PYTHON}" "${SUMMARY_SCRIPT}" \
    --dataset-root "${DATASET_ROOT}" \
    --model-glob "${MODEL_GLOB}" \
    --output "${OUTPUT_ROOT}/DA3_ARTIFACTS_PROGRESS.md" >/dev/null
}

case "${RUN_GROUP}" in
  vlm)
    echo "阶段 2/5：独立运行 2 项远程 VLM 指标"
    run_jobs "VLM" "${VLM_JOBS[@]}"
    ;;
  non_vlm)
    echo "阶段 2/5：独立运行 6 项本地非 VLM 指标"
    run_jobs_parallel "非 VLM" "${NON_VLM_JOBS[@]}"
    ;;
  non_da3)
    echo "阶段 2/5：独立运行 8 项非 DA3 指标"
    run_jobs_parallel "非 DA3" "${NON_DA3_JOBS[@]}"
    ;;
  da3)
    echo "阶段 2/5：独立准备并校验 DA3 artifacts"
    run_da3_stage
    echo "阶段 3/5：运行 4 项 DA3 依赖指标"
    run_jobs "DA3 依赖" "${DA3_DEPENDENT_JOBS[@]}"
    ;;
  all)
    echo "阶段 2/5：后台启动 DA3，同时运行非 DA3 指标"
    (
      set -o pipefail
      run_da3_stage
    ) &
    DA3_PID=$!
    run_jobs "与 DA3 并行" "${NON_DA3_JOBS[@]}"
    if ! wait "${DA3_PID}"; then
      DA3_PID=""
      echo "后台 DA3 stage 失败；已完成的非 DA3 指标会保留。" >&2
      exit 2
    fi
    DA3_PID=""
    run_jobs "DA3 依赖" "${DA3_DEPENDENT_JOBS[@]}"
    ;;
esac

echo "阶段 5/5：写入完成状态表"
REPORT="${OUTPUT_ROOT}/FULL_EVALUATION_STATUS.md"
{
  echo "# MBench-A 全量评测状态"
  echo
  echo "| subset | metric | 状态 | 输出目录 |"
  echo "|---|---|---|---|"
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r subset metric kind <<<"${spec}"
    short_name="${metric#mbencha.}"
    metric_dir="${OUTPUT_ROOT}/metrics/${short_name}"
    status="未完成"
    [[ -f "${metric_dir}/.complete" ]] && status="完成"
    echo "| ${subset} | ${metric} | ${status} | metrics/${short_name} |"
  done
} >"${REPORT}"

echo "MBench-A 全量评测完成：${REPORT}"
