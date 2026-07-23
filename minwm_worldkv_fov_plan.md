# minWM + WorldKV + FOV 检索路线规划

## 运行参数速查

下面这些参数都可以作为环境变量写在命令前面，例如：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

SAMPLES_PER_BUCKET=8 \
NUM_OUTPUT_FRAMES=120 \
SEEDS="0" \
CASES="baseline fixed_sink periodic_sink pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store" \
RUN_ROOT=./outputs/viewbench_large24_official_vbench_120f \
bash Wan21/scripts/inference/run_viewbench_large_with_official_vbench.sh
```

### 大规模 ViewBench + 官方 VBench 一体化脚本

入口：

```bash
bash Wan21/scripts/inference/run_viewbench_large_with_official_vbench.sh
```

这个脚本会先构造 ViewBench prompt/pose 子集，然后运行实验矩阵，并在每个 case 生成完成后自动调用官方 VBench 评测。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `PYTHON_BIN` | `/pool/hdd/home/hhhjyz/miniconda3/envs/ling/bin/python` | minWM 侧脚本使用的 Python，一般用 `ling` 环境。 |
| `VIEWBENCH_DATASET_ROOT` | `/pool/hdd/home/hhhjyz/research/datasets/ViewBench4Training` | 已解压的 ViewBench 数据目录。 |
| `OFFICIAL_VBENCH_ROOT` | `/pool/hdd/home/hhhjyz/research/VBench` | 官方 VBench 仓库路径。 |
| `OFFICIAL_VBENCH_PYTHON` | `/home/hhhjyz/miniconda3/envs/vbench/bin/python` | 官方 VBench 独立环境 Python。 |
| `SAMPLES_PER_BUCKET` | `8` | 每类 ViewBench 采样数量；总样本数约为 `3 * SAMPLES_PER_BUCKET`。 |
| `VIEWBENCH_MAX_POSE_FRAMES` | `360` | 构造 pose 子集时每条轨迹最多保留多少帧。 |
| `VIEWBENCH_PROMPT_DIR` | `Wan21/prompts/viewbench_large_$((SAMPLES_PER_BUCKET * 3))` | 生成的 prompt/pose/manifest 保存目录。 |
| `SKIP_BUILD_VIEWBENCH_PROMPTS` | `0` | 设为 `1` 时跳过数据子集构造，直接复用已有 `VIEWBENCH_PROMPT_DIR`。 |
| `RUN_ROOT` | `./outputs/viewbench_large_${SAMPLES_PER_BUCKET}x3_<timestamp>` | 本次实验输出根目录。 |
| `MAX_PROMPTS` | `3 * SAMPLES_PER_BUCKET` | 本次实际推理的 prompt 数量。 |
| `NUM_OUTPUT_FRAMES` | `120` | 每条视频生成帧数；建议使用 4 的倍数。 |
| `SEEDS` | `0` | 随机种子列表，例如 `"0 1 2"`。 |
| `CASES` | `baseline fixed_sink periodic_sink pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store` | 要运行的实验 case。 |
| `OFFICIAL_VBENCH_DIMENSIONS` | `subject_consistency background_consistency temporal_flickering motion_smoothness aesthetic_quality imaging_quality` | 官方 VBench 维度。默认不跑 `dynamic_degree`。 |
| `EVAL_ENABLE` | `1` | 是否启用评测阶段。 |
| `EVAL_DEVICE` | `cuda` | ViewBench/轻量评测使用设备。 |
| `EVAL_VBENCH_STYLE_ENABLE` | `0` | 是否额外跑本项目的 VBench-style proxy 指标。 |
| `EVAL_OFFICIAL_VBENCH_ENABLE` | `1` | 是否自动跑官方 VBench。 |
| `CONTINUE_ON_ERROR` | `1` | 单个 case 失败后是否继续跑后续 case。 |
| `DRY_RUN` | `0` | 设为 `1` 时只打印将要运行的命令，不真正推理。 |
| `SKIP_COMPLETED` | `1` | 设为 `1` 时跳过已完成推理和已有评测结果，适合断点续跑。 |

推荐大规模实验命令：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

SAMPLES_PER_BUCKET=8 \
NUM_OUTPUT_FRAMES=120 \
SEEDS="0" \
CASES="baseline fixed_sink periodic_sink pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store" \
RUN_ROOT=./outputs/viewbench_large24_official_vbench_120f \
SKIP_COMPLETED=1 \
bash Wan21/scripts/inference/run_viewbench_large_with_official_vbench.sh
```

如果想扩大到 48 条样本：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

SAMPLES_PER_BUCKET=16 \
NUM_OUTPUT_FRAMES=120 \
SEEDS="0" \
CASES="baseline fixed_sink periodic_sink pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store" \
RUN_ROOT=./outputs/viewbench_large48_official_vbench_120f \
SKIP_COMPLETED=1 \
bash Wan21/scripts/inference/run_viewbench_large_with_official_vbench.sh
```

输出文件：

```text
RUN_ROOT/experiment_manifest.txt
RUN_ROOT/experiment_summary.tsv
RUN_ROOT/seed_<seed>/<case>/inference_times.csv
RUN_ROOT/seed_<seed>/<case>/eval/official_vbench_metrics.csv
RUN_ROOT/official_vbench_report/official_vbench_summary.csv
RUN_ROOT/official_vbench_report/official_vbench_report.tex
```

### 实验矩阵脚本

入口：

```bash
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

这个脚本不构造数据集，直接使用已有 prompt/pose/manifest 运行实验矩阵。它也支持生成后自动评测。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `RUN_ROOT` | `./outputs/viewbench_worldkv_fov_<timestamp>` | 输出根目录。 |
| `DATA_PATH` | `Wan21/prompts/viewbench_small12/prompts.txt` | prompt 文件。 |
| `TRAJECTORY_POSE_PATH` | `Wan21/prompts/viewbench_small12/trajectory_pose_paths.txt` | pose `.npz` 路径列表。 |
| `VIEWBENCH_MANIFEST` | `Wan21/prompts/viewbench_small12/manifest.json` | ViewBench manifest，用于参考指标。 |
| `CHECKPOINT_PATH` | `/pool/hdd/home/hhhjyz/research/ckpts/Wan21/Action2V/dmd/model.pt` | minWM DMD checkpoint。 |
| `CONFIG_PATH` | `Wan21/configs/causal_forcing_dmd_camera.yaml` | 推理配置。 |
| `PYTHON_BIN` | `python`，找不到时 fallback 到 `python3` | minWM 侧评测脚本 Python。 |
| `MAX_PROMPTS` | `5` | 推理样本数。 |
| `PROMPT_START` | `0` | 从第几个 prompt 开始。 |
| `NUM_OUTPUT_FRAMES` | `120` | 每条视频生成帧数。 |
| `SEEDS` | `0` | seed 列表。 |
| `CASES` | 默认 5 组主实验 | 指定 case 列表。 |
| `DRY_RUN` | `0` | 只打印命令，不运行。 |
| `SKIP_COMPLETED` | `0` | 跳过已有完整推理和评测；大规模 wrapper 默认设为 `1`。 |
| `CONTINUE_ON_ERROR` | `1` | 失败后继续后续 case。 |

可用 case：

```text
baseline
fixed_sink
periodic_sink
bank_random_sink
bank_uniform_sink
bank_pose_sink
bank_worldkv_fov_sink
kv_bank_only
pose
pose_rope
worldkv_fov
hy_fov
hybrid
pose_compress_store
worldkv_fov_compress_store
worldkv_fov_dynamic_compress_store
```

### Sink / KV Bank / Retrieval 参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `SINK_SIZE` | `4` | sink frame 数。 |
| `SINK_UPDATE_INTERVAL` | `4` | periodic 或 bank sink 每隔多少 block 更新一次。 |
| `SINK_BANK_SEED` | `0` | bank random sink 的随机种子。 |
| `KV_BANK_DEVICE` | `cpu` | KV bank 存储设备。 |
| `KV_BANK_MAX_BLOCKS` | `45` | KV bank 最多保留多少 block。 |
| `KV_BANK_WARN_MEMORY_GB` | `128` | KV bank 内存告警阈值。 |
| `KV_BANK_LOG_INTERVAL` | `1` | KV bank 日志间隔。 |
| `RETRIEVAL_FRAMES` | `12` | 每次检索取回多少历史 frame。 |
| `RETRIEVAL_RECENT_FRAMES` | `32` | 检索时排除最近多少 frame，避免只取近邻。 |
| `RETRIEVAL_FOV_SAMPLES` | `8192` | FOV overlap 采样点数。 |
| `RETRIEVAL_FOV_RADIUS` | `8.0` | FOV overlap 几何采样半径。 |
| `RETRIEVAL_FOV_H_DEG` | `60.0` | 水平视场角。 |
| `RETRIEVAL_FOV_V_DEG` | `35.0` | 垂直视场角。 |
| `RETRIEVAL_HYBRID_FOV_WEIGHT` | `0.5` | hybrid metric 中 FOV 分量权重。 |
| `PROPE_REENCODE_MODE` | `none` | prope 重编码策略。 |

### KV Compression / 动态剪枝参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `KV_COMPRESSION_KEEP_RATIO` | `0.5` | 静态压缩保留比例。 |
| `KV_COMPRESSION_DYNAMIC_MIN_KEEP` | `0.25` | 动态剪枝最小保留比例。 |
| `KV_COMPRESSION_DYNAMIC_MAX_KEEP` | `0.75` | 动态剪枝最大保留比例。 |
| `KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE` | `1.0` | 平移幅度归一化尺度。 |
| `KV_COMPRESSION_DYNAMIC_ROTATION_SCALE` | `0.35` | 旋转幅度归一化尺度，单位 rad。 |
| `KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT` | `0.25` | 相机运动幅度对 keep ratio 的影响权重。 |

### Profiling / 评测参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `LOG_CACHE_STATE` | `0` | 是否记录 cache/KV 状态。 |
| `LOG_CACHE_INTERVAL` | `1` | cache 状态日志间隔。 |
| `EVAL_ENABLE` | `1` | 是否启用评测。 |
| `EVAL_DEVICE` | `cuda` | 轻量评测设备。 |
| `EVAL_PAIR_RADIUS` | `0` | ViewBench reference pair 半径。 |
| `EVAL_VBENCH_STYLE_ENABLE` | `1` | 是否运行项目内 VBench-style proxy；大规模 wrapper 默认关。 |
| `EVAL_VBENCH_STYLE_MAX_FRAMES` | `96` | VBench-style 最多采样帧数。 |
| `EVAL_VBENCH_STYLE_RESIZE_WIDTH` | `256` | VBench-style 评测 resize 宽度。 |
| `EVAL_OFFICIAL_VBENCH_ENABLE` | `0` | 是否运行官方 VBench；大规模 wrapper 默认开。 |
| `OFFICIAL_VBENCH_ROOT` | 空 | 官方 VBench 仓库路径。 |
| `OFFICIAL_VBENCH_PYTHON` | `python` | 官方 VBench 环境 Python。 |
| `OFFICIAL_VBENCH_LONG` | `0` | 是否使用 VBench-Long 入口。 |
| `OFFICIAL_VBENCH_DIMENSIONS` | 空，使用 adapter 默认维度 | 官方 VBench 维度列表。 |

## 目标

建立一条适合单卡 A100 40G 的 world memory 实验路线：

1. 先跑通 minWM 的 Wan21 Action2V。
2. 在 minWM 中研究 sink frame 的保留与定时更新策略。
3. 在 minWM 中加入 WorldKV 风格的 KV bank。
4. 对比不同历史检索算法：recent-only、pose retrieval、FOV retrieval、hybrid retrieval。
5. 基于相机运动幅度或 FOV overlap，动态调整 KV 剪枝率。

短期目标不是直接复现 WorldKV 14B，而是使用 minWM 的 1.3B Wan Action2V 作为可运行、可迭代的实验平台，然后逐步加入 WorldKV 风格的 memory 机制。

## 涉及目录

- `minWM/`
  - 主要实验平台。
  - Wan21 Action2V 是 1.3B，支持相机控制、causal/autoregressive 推理，更适合单卡 A100 40G。
- `WorldKV/`
  - KV-bank retrieval、retrieval metric、KV compression、attention window 设计的参考实现。
- `HY-WorldPlay/`
  - FOV overlap retrieval 的几何参考实现。

## 当前理解

### minWM 的优势

- Wan21 Action2V 使用 Wan2.1-T2V-1.3B 作为 base。
- 已经支持 camera trajectory control。
- causal inference 已经使用 KV cache。
- causal attention 代码中已经有 `sink_size` 概念。
- 默认 camera DMD 配置中包含：

```yaml
num_frame_per_block: 4
model_kwargs:
  local_attn_size: 20
  use_camera: true
```

也就是说，minWM 默认以 4 个 latent frame 为一个 block 推理，local attention window 为 20 个 latent frames。

### WorldKV 的优势

WorldKV 提供了清晰的 memory 窗口结构：

```text
[sink | retrieved history | recent]
```

它已经包含：

- chunk-level KV bank
- pose-based retrieval
- anchor + novelty KV pruning
- retrieval-time / store-time compression 思路

我们此前也已经在 `WorldKV/wan/modules/retrieval_utils.py` 中加入了 FOV/hybrid retrieval metric，可作为后续移植参考。

### HY-WorldPlay 的优势

HY-WorldPlay 提供了基于相机几何的 FOV overlap retrieval 逻辑。它适合作为算法参考，但不应该直接照搬 W2C/C2W 坐标处理到 minWM；minWM 和 WorldKV 应该使用各自原生的 pose convention。

## 里程碑 0：跑通 minWM baseline

### 目标

在不修改代码的情况下跑通官方 Wan21 Action2V causal camera demo。

### 命令

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

# 如果当前环境没有 modelscope CLI，先安装：
# pip install modelscope

modelscope download \
  --model Wan-AI/Wan2.1-T2V-1.3B \
  --local_dir ./ckpts/Wan2.1-T2V-1.3B

mkdir -p Wan21/wan_models
ln -s "$(realpath ./ckpts/Wan2.1-T2V-1.3B)" Wan21/wan_models/Wan2.1-T2V-1.3B

# 注意：截至 2026-07-14，MIN-Lab/minWM 没有可直接访问的 ModelScope 镜像。
# minWM 的训练好 checkpoint 仍需从 Hugging Face 下载；国内网络可使用 HF_ENDPOINT 镜像。
HF_ENDPOINT=https://hf-mirror.com \
huggingface-cli download MIN-Lab/minWM \
  --local-dir ./ckpts \
  --include "Wan21/Action2V/dmd/*"

CHECKPOINT_PATH=../ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=./outputs/quickstart_wan_action2v \
TRAJECTORY_PATH="../Wan21/prompts/trajectories.txt" \
bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

### ViewBench small12 小样本 smoke test

完整 `demos.txt` 当前有 30 条 prompt，逐条全部推理会很慢。为了研究 sink frame、KV cache、WorldKV/FOV retrieval 对长视频 loop closure 的影响，默认改为使用 ViewBench public split 构造 `viewbench_small12`：

- 4 条纯旋转。
- 4 条多轴旋转。
- 4 条旋转+平移。
- 每条使用 ViewBench 原始 per-frame pose，保存为 minWM 可直接读取的 `.npz`。

先下载并解压 ViewBench：

```bash
cd /pool/hdd/home/hhhjyz/research

modelscope download \
  --dataset JEdward/viewbench-dataset \
  --local_dir ./datasets/ViewBench-v1

mkdir -p ./datasets/ViewBench4Training
for shard in ./datasets/ViewBench-v1/pure_rotation_*.tar.zst ./datasets/ViewBench-v1/rotation_translation_*.tar.zst; do
  tar --zstd -xf "$shard" -C ./datasets/ViewBench4Training
done
```

构造 minWM 小测试集：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

python Wan21/scripts/data_preprocessing/build_viewbench_small.py \
  --viewbench-root /pool/hdd/home/hhhjyz/research/datasets/ViewBench4Training \
  --output-dir Wan21/prompts/viewbench_small12 \
  --max-frames 360
```

生成文件：

```text
Wan21/prompts/viewbench_small12/prompts.txt
Wan21/prompts/viewbench_small12/trajectory_pose_paths.txt
Wan21/prompts/viewbench_small12/poses/*.npz
Wan21/prompts/viewbench_small12/manifest.json
```

运行 smoke test：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

### 带耗时 profiling 的 smoke test

如果需要在推理结束后自动保存每个视频的耗时，使用 profiling 入口：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

MAX_PROMPTS=5 \
NUM_OUTPUT_FRAMES=180 \
OUTPUT_FOLDER=./outputs/profile_viewbench_small12 \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

生成内容会保存在：

```text
./outputs/profile_viewbench_small12/profile/
```

主要文件：

- `inference.log`：完整推理日志。
- `inference_times.csv`：每个视频的耗时明细。
- `profile_summary.json` / `profile_summary.txt`：平均每视频耗时、平均生成耗时、写视频耗时、wall time、last-chunk FPS 和峰值显存。
- `video_total_seconds.png`：每个视频总耗时图。
- `video_stage_seconds.png`：每个视频的 generation / postprocess / write video 分段耗时图。

默认不会记录 GPU 显存或 KV/cache 状态。如果某次确实要调试 cache，可以显式打开：

```bash
LOG_CACHE_STATE=1 \
MAX_PROMPTS=5 \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

如需覆盖默认值，可以在命令前设置环境变量：

```bash
PROMPT_START=0 \
MAX_PROMPTS=2 \
NUM_OUTPUT_FRAMES=180 \
OUTPUT_FOLDER=./outputs/viewbench_start0_n2_180 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

参数含义：

- `DATA_PATH=Wan21/prompts/viewbench_small12/prompts.txt`：默认使用 12 条 ViewBench 小样本 prompt。
- `TRAJECTORY_POSE_PATH=Wan21/prompts/viewbench_small12/trajectory_pose_paths.txt`：默认使用与 prompt 对齐的 ViewBench pose `.npz`。
- `MAX_PROMPTS=1`：默认只跑 1 条 prompt，避免第一次 smoke test 太慢；正式小规模比较建议设为 12。
- `PROMPT_START=0`：从第 0 条 prompt 开始；例如设为 `5` 就跑第 5 条开始的样本。
- `NUM_OUTPUT_FRAMES=180`：默认生成 180 帧，方便观察长时一致性。causal camera 默认 `num_frame_per_block=4`，建议使用 4 的倍数；如果设为 `0` 或负数，会使用每条 pose `.npz` 的原生长度。
- `TRAJECTORY_POSE_PATH` 仍然使用完整 pose list 文件；代码会按原始 prompt 下标自动取对应行。

ViewBench small12 的构造脚本和说明：

- `minWM/Wan21/scripts/data_preprocessing/build_viewbench_small.py`
- `minWM/Wan21/prompts/viewbench_small12/README.md`

如果只想临时绕过代码参数，也可以手动切小文件：

```bash
head -n 1 Wan21/prompts/demos.txt > Wan21/prompts/demos_1.txt
head -n 1 Wan21/prompts/trajectories.txt > Wan21/prompts/trajectories_1.txt
```

### 需要记录

- 峰值显存。
- 每个 prompt 的推理时间。
- last-chunk FPS。
- 输出视频质量。
- camera trajectory 是否明显生效。
- 是否有依赖、checkpoint 路径或权重加载问题。

### ViewBench small12 评测

#### 推荐命令：已有结果单独评测

如果已经有某个 case 的推理结果，例如：

```text
./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/
```

先在 `ling` 环境下跑 loop-closure 指标：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

python Wan21/scripts/evaluation/evaluate_viewbench_small.py \
  --manifest Wan21/prompts/viewbench_small12/manifest.json \
  --timing-csv ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/inference_times.csv \
  --output-dir ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/eval \
  --device cuda
```

再跑 VBench-style 无参考视频质量 proxy：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

python Wan21/scripts/evaluation/evaluate_vbench_style.py \
  --timing-csv ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/inference_times.csv \
  --output-dir ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/eval \
  --max-frames 96 \
  --resize-width 256
```

输出：

- `viewbench_metrics.csv`：每个视频的 PSNR、SSIM、LPIPS、last-chunk FPS、峰值显存。
- `viewbench_metrics.json`：平均指标和评测说明。
- `vbench_style_metrics.csv`：每个视频的无参考视频质量指标。
- `vbench_style_metrics.json`：平均无参考指标和解释说明。

LPIPS 依赖 `lpips` 包；如果当前环境没有安装，脚本会跳过 LPIPS 并保留 PSNR/SSIM/FPS。FID 暂不在轻量评测中计算，因为没有明确参考分布时，FID 对当前生成式 loop-closure 对比的解释性较弱。

#### 推荐命令：推理后自动评测

矩阵脚本 `run_viewbench_worldkv_fov_experiments.sh` 默认会在每个 case 推理成功后自动运行：

- `evaluate_viewbench_small.py`
- `evaluate_vbench_style.py`

因此最常用的一键命令是：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

RUN_ROOT=./outputs/viewbench_worldkv_fov_a10080_five_cases_120f \
SEEDS="0" \
MAX_PROMPTS=12 \
NUM_OUTPUT_FRAMES=120 \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

如果只想重新跑推理、不跑任何评测：

```bash
EVAL_ENABLE=0 \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

如果只想关闭 VBench-style proxy，但保留 ViewBench loop-closure 指标：

```bash
EVAL_VBENCH_STYLE_ENABLE=0 \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

VBench-style 轻量评测不使用 VBench 官方 prompt，也不声称是官方 leaderboard 分数。它直接读取我们自己的 `inference_times.csv` 和生成视频，计算不依赖文本 prompt 的内部对比指标。当前包含：

- `frame_difference_mean`：相邻帧平均变化，反映整体动态程度。
- `temporal_flicker_proxy`：二阶帧差，越低通常越稳定。
- `motion_smoothness_proxy`：由 flicker proxy 转换得到，越高越平滑。
- `dynamic_degree_proxy` / `optical_flow_mean`：稠密光流均值，作为动态程度近似。
- `sharpness_laplacian_var`：Laplacian 方差，作为清晰度近似。
- `brightness_mean`、`contrast_mean`、`black_frame_ratio`、`white_frame_ratio`：基础画面质量诊断。

#### 可选命令：官方 VBench 质量维度

如果已经安装官方 VBench，也可以使用官方指标入口。推荐先只跑 custom-input 较稳的质量维度，不需要 VBench prompt，也暂时不需要 detectron2：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

python Wan21/scripts/evaluation/evaluate_vbench_official.py \
  --timing-csv ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/inference_times.csv \
  --output-dir ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/eval \
  --vbench-root /path/to/VBench \
  --python-bin /home/hhhjyz/miniconda3/envs/vbench/bin/python \
  --dimensions subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality
```

VBench-Long：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

python Wan21/scripts/evaluation/evaluate_vbench_official.py \
  --timing-csv ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/inference_times.csv \
  --output-dir ./outputs/viewbench_worldkv_fov_a10080_five_cases_120f/seed_0/baseline/eval \
  --vbench-root /path/to/VBench \
  --python-bin /home/hhhjyz/miniconda3/envs/vbench/bin/python \
  --long \
  --dimensions subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality
```

如果希望矩阵脚本在每个 case 推理后自动调用官方 VBench：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

EVAL_OFFICIAL_VBENCH_ENABLE=1 \
OFFICIAL_VBENCH_ROOT=/path/to/VBench \
OFFICIAL_VBENCH_PYTHON=/home/hhhjyz/miniconda3/envs/vbench/bin/python \
OFFICIAL_VBENCH_DIMENSIONS="subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality" \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

默认官方 VBench 入口是关闭的，因为它需要额外依赖和模型权重。当前 Forcing-KV 的 `evaluation/VBench` 目录只是 wrapper 和总分脚本，不包含完整官方 VBench 仓库。

官方 VBench 依赖建议单独建环境，不建议直接塞进当前 `ling` 推理环境：

```bash
conda create -n vbench python=3.10 -y
conda activate vbench

# 官方 README 推荐 CUDA <= 12.1 的 PyTorch 组合；示例使用 cu118。
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install vbench

git clone https://github.com/Vchitect/VBench.git /path/to/VBench
```

`detectron2` 不是当前推荐质量维度的必需项，可以先不装。只有当你要跑对象/场景/语义检测类维度时再安装。若安装时报 `No module named 'torch'`，先确认当前 `vbench` 环境里能 `import torch`，然后使用：

```bash
conda activate vbench
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install --no-build-isolation 'detectron2@git+https://github.com/facebookresearch/detectron2.git'
```

当前优先推荐官方 custom input 可较稳使用的维度：

```text
subject_consistency
background_consistency
motion_smoothness
dynamic_degree
aesthetic_quality
imaging_quality
```

如果跑普通 VBench 并希望包含 `temporal_flickering`，可以额外加入：

```bash
OFFICIAL_VBENCH_DIMENSIONS="subject_consistency background_consistency temporal_flickering motion_smoothness dynamic_degree aesthetic_quality imaging_quality"
```

注意：`object_class`、`multiple_objects`、`color`、`spatial_relationship`、`scene`、`human_action` 等语义维度更依赖 VBench prompt suite 或 prompt 元数据。对我们自己的 ViewBench/minWM prompt 可以计算，但解释时只能作为内部参考，不能直接等同官方 leaderboard。

### 成功标准

- 至少生成一个视频。
- 相机轨迹对生成结果有可见影响。
- 单卡 A100 40G 不 OOM。

## 里程碑 1：加入 baseline 日志

### 目标

在改变 memory 行为之前，先记录足够的运行状态，方便后续定位问题。

### 当前状态

已实现。默认关闭，可以通过推理脚本环境变量打开：

```bash
LOG_CACHE_STATE=1 \
LOG_CACHE_INTERVAL=1 \
CHECKPOINT_PATH=/pool/hdd/home/hhhjyz/research/ckpts/Wan21/Action2V/dmd/model.pt \
OUTPUT_FOLDER=./outputs/cache_log_test \
TRAJECTORY_PATH="Wan21/prompts/trajectories.txt" \
bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

`LOG_CACHE_INTERVAL=N` 表示每 N 个生成 block 打一次日志。

也可以直接调用 `Wan21/wan_inference.py` 时使用：

```bash
--log_cache_state --log_cache_interval 1
```

日志格式示例：

```text
[cache-state] tag=before_denoise block=0 frames=0:4 tokens=0:6240 kv=(local_end=0 global_end=0 capacity=31200) prope=(...) pose_frames=4 pose_t_norm=... mem_alloc=...GB mem_reserved=...GB mem_peak=...GB
```

### 建议记录内容

每个生成 block 记录：

- `current_start_frame`
- `current_num_frames`
- `kv_cache local_end_index`
- `kv_cache global_end_index`
- `prope_kv_cache local_end_index`
- CUDA allocated/reserved memory
- 当前 trajectory chunk id 和 pose 摘要

### 代码切入点

- `minWM/Wan21/pipeline/causal_inference.py`
- `minWM/Wan21/pipeline/causal_diffusion_inference.py`
- `minWM/Wan21/wan/modules/causal_model.py`

### 成功标准

- 日志能清楚展示 cache 随 block 推进的变化。
- 日志可以通过配置或参数关闭，避免污染正常推理输出。
- `run_infer_causal_camera.sh` 和 `run_infer_ar_camera.sh` 都支持 `LOG_CACHE_STATE` / `LOG_CACHE_INTERVAL`。

## 里程碑 2：sink frame 策略实验

### 目标

在加入完整 retrieval 之前，先测试长期 memory anchor 是否能改善长时一致性。

### 当前状态

已实现，默认关闭，不改变 baseline。新增参数：

```bash
--sink_strategy none|fixed|periodic
--sink_size 4
--sink_update_interval 4
```

推理脚本环境变量：

```bash
SINK_STRATEGY=none|fixed|periodic
SINK_SIZE=4
SINK_UPDATE_INTERVAL=4
```

实现位置：

- `minWM/Wan21/wan_inference.py`
- `minWM/Wan21/pipeline/sink_utils.py`
- `minWM/Wan21/pipeline/causal_inference.py`
- `minWM/Wan21/pipeline/causal_diffusion_inference.py`
- `minWM/Wan21/scripts/inference/run_sink_ablation_causal_camera.sh`

### 实验变体

1. 原始 minWM：

```text
sink_size = 0
local_attn_size = 20
```

2. 固定初始 sink：

```text
sink_size = 4
local_attn_size = 20
```

含义是：最开始的 4 个 latent frames 永久保留在 cache 前部，其余 cache 正常滚动。

3. 定时更新 sink：

```text
sink_size = 4
每 N 个 block 更新一次 sink
```

含义是：每隔 N 个生成 block，把最新 clean block 的 KV 写入 sink 区域，替换旧的 sink。

### 推荐运行命令

一键跑 baseline / fixed sink / periodic sink 三组对比：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

MAX_PROMPTS=5 \
NUM_OUTPUT_FRAMES=180 \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=4 \
RUN_PREFIX=sink_ablation_viewbench180 \
bash Wan21/scripts/inference/run_sink_ablation_causal_camera.sh
```

输出目录示例：

```text
outputs/sink_ablation_viewbench180_baseline_0_5_180/
outputs/sink_ablation_viewbench180_fixed_sink4_0_5_180/
outputs/sink_ablation_viewbench180_periodic_sink4_int4_0_5_180/
```

每个目录下都会保留：

- 生成视频；
- `inference_times.csv` / `inference_times.json`；
- `profile/profile_summary.txt`；
- `profile/video_total_seconds.png`；
- `profile/video_stage_seconds.png`。

单独跑 fixed sink：

```bash
SINK_STRATEGY=fixed \
SINK_SIZE=4 \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=120 \
OUTPUT_FOLDER=./outputs/fixed_sink4_test \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

单独跑 periodic sink：

```bash
SINK_STRATEGY=periodic \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=4 \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=120 \
OUTPUT_FOLDER=./outputs/periodic_sink4_int4_test \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

### 关键风险

minWM 的 camera PRoPE 有独立的 `prope_kv_cache`。任何 sink 更新都必须同步处理：

- 普通 `kv_cache`
- `prope_kv_cache`

如果只更新其中一个，普通 attention 和 camera-aware PRoPE attention 会看到不一致的历史信息。

当前 periodic sink update 已同步处理：

- `kv_cache_pos` / `kv_cache_neg`
- `prope_kv_cache_pos` / `prope_kv_cache_neg`
- DMD/ODE 单条件路径中的 `kv_cache1` / `prope_kv_cache1`

当前实现是 `periodic sink v0`：直接复制已经编码好的 RoPE/PRoPE KV 到 sink 区域，不重新计算 RoPE 或 PRoPE 位置编码。这适合作为低成本 ablation，用来判断“定期刷新 sink memory”是否有实验价值，但它不是严格的位置校正或几何一致 replay。

后续做 WorldKV/FOV retrieval 级别的正式实现时，应升级为：

- 额外保存 raw K/V；
- 记录 frame id、block id、viewmats/Ks；
- replay 或 retrieval 时做 RoPE/PRoPE correction；
- 明确区分“memory slot 位置”和“原始相机/时间位置”。

### 实现提示

causal attention 的滚动逻辑已经会保护前面的 `sink_tokens`：

```python
sink_tokens = self.sink_size * frame_seqlen
```

所以 fixed sink 实验可以先从配置开始：

```yaml
model_kwargs:
  sink_size: 4
```

定时更新 sink 在 clean context pass 之后执行，把最新 block 的 clean KV 拷贝到 sink token 区域。该操作不改变 cache index，只替换 sink 区域内容，因此可以和原有 local window 滚动逻辑共存。

### 成功标准

- fixed sink 不改变输出 tensor 形状，不引入 cache index 错误。
- periodic sink update 能完整跑完推理。
- 可以和原始 baseline 使用同一 prompt、trajectory、seed 做视频对比。

## 里程碑 3：在 minWM 中建立 KV bank（已完成）

### 目标

保存每个生成 block 在 clean context pass 后的历史 KV，但暂时不把 bank 接入 attention。这样可以独立验证存储逻辑、normal/PRoPE 对齐和内存开销，并保证关闭 retrieval 时不改变生成结果。

### 当前实现

核心实现：

- `Wan21/pipeline/kv_bank.py`
- `Wan21/pipeline/causal_inference.py`：DMD/ODE 单条件路径，保存 `main` 分支。
- `Wan21/pipeline/causal_diffusion_inference.py`：CFG diffusion 路径，分别保存 `cond` 和 `uncond` 分支。

每个 block 在 clean context pass 后执行：

```text
denoise 当前 block
写入 output
clean context pass 覆盖 cache 中当前 block 的 KV
从每层 cache 末尾抽取当前 block token
复制到 CPU/GPU KV bank
可选 periodic sink update
```

每个 block 保存：

- 每层 normal K/V。
- 每层 PRoPE K/V；没有 camera pose 时允许为空。
- `block_id`、`frame_start`、`frame_end`、`token_count`。
- 当前 block 的 `viewmats` 和 `Ks` CPU 副本。
- 平移幅度和旋转角 pose summary。
- block 与 bank 的精确存储字节数。

后续 retrieval 可以通过：

```python
pipeline.kv_bank.get_layer(block_index, layer_index, branch="main")
```

读取指定 block/layer/branch 的 KV。

### 配置参数

```text
KV_BANK_ENABLE=0             默认关闭
KV_BANK_DEVICE=cpu           默认使用 CPU 主存
KV_BANK_MAX_BLOCKS=0         0 表示不限制；正数表示 FIFO 保留上限
KV_BANK_LOG_INTERVAL=1       每 N 个 block 打印一次 bank 状态
KV_BANK_WARN_MEMORY_GB=16    预计存储超过该值时打印警告
```

对应 Python 参数：

```text
--kv_bank_enable
--kv_bank_device {cpu,cuda}
--kv_bank_max_blocks N
--kv_bank_log_interval N
--kv_bank_warn_memory_gb N
```

### 安全 smoke test

第一轮建议只保留 2 个 block：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

bash Wan21/scripts/inference/run_kv_bank_smoke_causal_camera.sh
```

等价参数：

```bash
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=20 \
KV_BANK_ENABLE=1 \
KV_BANK_DEVICE=cpu \
KV_BANK_MAX_BLOCKS=2 \
OUTPUT_FOLDER=./outputs/kv_bank_smoke \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

`inference_times.csv/json` 和 `profile_summary.json/txt` 新增：

- `kv_bank_blocks`
- `kv_bank_evicted_blocks`
- `kv_bank_total_bytes`
- `kv_bank_total_gb`
- `kv_bank_branches`

### 内存风险

未压缩 KV bank 非常大。Wan21 camera DMD 的一个 4-frame block 会保存 30 层 normal K/V 和 PRoPE K/V；diffusion 还会同时保存 cond/uncond 两套。运行 180 帧且 `KV_BANK_MAX_BLOCKS=0` 时可能消耗数十到上百 GB 主存。

因此当前建议：

- 功能验证使用 `KV_BANK_MAX_BLOCKS=1` 或 `2`。
- 正式验证完整增长前先观察日志中的 `block_gb` 和 `projected_gb`。
- A100 40G 上默认使用 CPU bank，不使用 GPU bank。
- 后续里程碑 6 加入 compression 后，再进行完整长视频 bank 实验。

### 当前限制

- bank 保存的是已经应用 RoPE/PRoPE 的 clean KV，不是 raw K/V。
- bank 当前不参与 attention，因此不会改善或损害生成画面，只增加复制耗时和内存占用。
- sequence parallel 模式下，每个 rank 保存本 rank 的 head shard；日志中的内存是单 rank 数值。
- `KV_BANK_MAX_BLOCKS>0` 时使用 FIFO 淘汰最旧 block。

### 成功标准

- `KV_BANK_ENABLE=0` 时不复制任何 bank tensor。
- 不设置 block 上限时，每生成一个 block，bank 长度增加 1。
- DMD 的 `main` 以及 diffusion 的 `cond/uncond` 都能保存 normal/PRoPE KV。
- bank 存储不进入 attention，不改变输出 tensor 形状和 cache index。
- CPU 内存增长可以通过日志和 timing 文件解释。

## 里程碑 4：加入 WorldKV 风格 retrieval window（第一版已完成）

### 目标

使用 KV bank 组成 WorldKV 风格 attention window：

```text
[sink | retrieved | recent]
```

### 第一版实现范围

- 默认关闭；通过 `--retrieval_enable` 或 `RETRIEVAL_ENABLE=1` 打开。
- 使用已有 KV bank 保存 clean context pass 后的 block KV。
- attention window 由原始 local recent window 改为可选的：

```text
[sink | retrieved | recent]
```

- DMD/ODE 路径使用 `main` 分支。
- diffusion CFG 路径分别使用 `cond` 和 `uncond` 分支。
- normal KV 和 PRoPE KV 都会构造 retrieval payload；如果没有 PRoPE payload，则 PRoPE 路径自动退回原 local window。
- KV bank 默认 CPU 存储，检索时按需搬到当前推理 device。
- 可选支持 WorldKV 风格 normal KV time-axis RoPE rebasing：

```bash
RETRIEVAL_ROPE_CORRECTION=1
```

该开关只修正 normal RoPE K；PRoPE KV 的几何位置重映射仍作为后续风险项保留。

### 代码切入点

当前 minWM attention 使用：

```python
attention(
    roped_query,
    kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
    kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
)
```

需要改成：

```python
k_cache = torch.cat([sink_k, retrieved_k, recent_k], dim=1)
v_cache = torch.cat([sink_v, retrieved_v, recent_v], dim=1)
attention(roped_query, k_cache, v_cache)
```

同样结构也要用于 `prope_kv_cache`。

当前代码位置：

- `Wan21/pipeline/kv_bank.py`
  - `KVBank.select_retrieval_blocks(...)`
  - `KVBank.get_retrieval_payloads(...)`
- `Wan21/pipeline/causal_inference.py`
  - DMD/ODE `main` 分支 retrieval payload 构造。
- `Wan21/pipeline/causal_diffusion_inference.py`
  - diffusion `cond/uncond` 分支 retrieval payload 构造。
- `Wan21/wan/modules/causal_model.py`
  - `_compose_attention_window(...)`
  - `CausalWanSelfAttention.forward(..., retrieval_kv=...)`
- `Wan21/wan_utils/wan_wrapper.py`
  - 将 `retrieval_kv` 透传到 causal model。

### 成功标准

- retrieval 可以通过配置启用/关闭。
- retrieved block 数为 0 时，行为应退化为原始 local window 或 sink-only。
- retrieved window token 数稳定。

## 里程碑 5：比较不同检索算法（第一版已完成）

### 目标

加入 retrieval metric 选择：

```text
recent_only
pose
worldkv_fov
hy_fov
hybrid
```

### 算法定义

#### Recent Only

不检索历史，只使用 local recent window。作为最基础 baseline。

#### Pose Retrieval

参考 WorldKV：

```text
translation distance + rotation geodesic distance
```

#### FOV Retrieval

当前实现了两种 FOV 检索：

1. `worldkv_fov`
   - 参考 WorldKV 的 C2W deterministic probe points。
   - 使用当前 block 和历史 block 的中间帧估计 FOV overlap。
2. `hy_fov`
   - 参考 HY-WorldPlay 的 W2C angular FOV overlap。
   - 使用 pitch/yaw + spherical probe 判断视锥重叠。

```text
distance = 1 - FOV_overlap(current block, historical block)
```

注意：minWM 中 `viewmats` 按 W2C 存储；WorldKV-style FOV 会显式转为 C2W，HY-style FOV 保留 W2C angular convention。

#### Hybrid Retrieval

```text
distance = (1 - alpha) * normalized_pose_distance + alpha * fov_distance
```

### 需要记录的诊断信息

每个 block 记录：

- candidate block ids
- selected block ids
- pose distances
- FOV overlap scores
- final retrieval distance
- retrieval 额外耗时
- 峰值显存

第一版目前已经支持 metric 切换，并加入了 block-level retrieval diagnostics。每个视频推理结束后，会在 `OUTPUT_FOLDER` 下追加：

```text
retrieval_events.csv
retrieval_events.jsonl
```

如果使用 profiled 入口，这两个文件也会复制到：

```text
OUTPUT_FOLDER/profile/retrieval_events.csv
OUTPUT_FOLDER/profile/retrieval_events.jsonl
```

主要字段：

- `sample_order` / `prompt_index`
- `branch`：DMD 为 `main`，diffusion 为 `cond` 或 `uncond`
- `current_frame_start` / `current_num_frames`
- `metric`
- `candidate_block_ids`
- `selected_block_ids`
- `selected_frame_starts`
- `distances`
- `retrieved_tokens_per_layer`
- `selection_seconds`
- `payload_seconds`

### 成功标准

- 不同 metric 能够选择不同的历史 block。
- 同一 seed 下检索结果可复现。
- 日志足够解释视频中的回环、一致性或漂移现象。

## 里程碑 6：加入 KV compression（第一版已完成）

### 目标

在 attention 前减少 retrieved KV token 数，降低显存和计算开销。

### 第一版方案

移植 WorldKV 的 anchor + novelty pruning：

```text
anchor frame 完整保留
非 anchor frame 中，保留和 anchor centroid 最不相似的 token
```

### minWM 特有注意点

compression 必须一致应用到：

- normal KV
- PRoPE KV

当前第一版会对 normal KV 和 PRoPE KV 应用相同的 anchor/novelty 规则，但每条张量路径各自根据自己的 K 计算 token novelty。后续如果要更严格对齐 ordinary attention 和 PRoPE attention，应改成共享同一组 token indices。

支持两种模式：

- store-time compression：`--kv_compression_enable --kv_compression_at_store`
- retrieval-time compression：`--kv_compression_enable` 且不设置 `--kv_compression_at_store`

相关参数：

```bash
--kv_compression_keep_ratio 0.5
--kv_compression_anchor_rotate
--kv_compression_pooled
```

### 成功标准

- compression 改变 token 数，但不改变 batch/head/dim 维度。
- compression 可以关闭。
- 分别测试 store-time compression 和 retrieval-time compression。

### 当前 smoke 命令

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

RETRIEVAL_ENABLE=1 \
RETRIEVAL_METRIC=pose \
RETRIEVAL_FRAMES=4 \
RETRIEVAL_RECENT_FRAMES=4 \
RETRIEVAL_ROPE_CORRECTION=1 \
KV_BANK_DEVICE=cpu \
KV_BANK_MAX_BLOCKS=2 \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=20 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

测试 WorldKV FOV：

```bash
RETRIEVAL_ENABLE=1 \
RETRIEVAL_METRIC=worldkv_fov \
RETRIEVAL_FRAMES=4 \
RETRIEVAL_FOV_SAMPLES=2048 \
KV_BANK_MAX_BLOCKS=2 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

测试 HY-WorldPlay FOV：

```bash
RETRIEVAL_ENABLE=1 \
RETRIEVAL_METRIC=hy_fov \
RETRIEVAL_FRAMES=4 \
RETRIEVAL_FOV_SAMPLES=2048 \
KV_BANK_MAX_BLOCKS=2 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

测试 store-time compression：

```bash
RETRIEVAL_ENABLE=1 \
RETRIEVAL_METRIC=pose \
RETRIEVAL_FRAMES=4 \
KV_COMPRESSION_ENABLE=1 \
KV_COMPRESSION_AT_STORE=1 \
KV_COMPRESSION_KEEP_RATIO=0.5 \
KV_BANK_MAX_BLOCKS=2 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

## 里程碑 7：基于相机运动动态调整剪枝率

### 目标

根据相机运动幅度或 FOV overlap 动态调整 retrieved KV 的 keep ratio。

### 可用信号

1. 当前 block 与 retrieved block 的平移幅度。
2. 当前 block 与 retrieved block 的旋转角。
3. FOV overlap score。
4. pose + FOV 的混合 camera-motion score。

### 简单离散策略

```text
if fov_overlap > 0.6:
    keep_ratio = 0.75
elif fov_overlap > 0.3:
    keep_ratio = 0.50
else:
    keep_ratio = 0.25
```

### 连续策略

```text
motion_score = a * translation_norm + b * rotation_angle_norm
keep_ratio = clamp(base + c * fov_overlap - d * motion_score, min_keep, max_keep)
```

### 成功标准

- dynamic pruning 根据每个 stored block 的相机运动幅度改变实际 keep ratio，并降低平均 retrieved token 数。
- 相比固定 keep ratio，不明显损害 camera-following 稳定性。
- 在相近视觉质量下提升速度或降低显存。

### 当前实现状态

第一版动态剪枝已经接入 store-time KV compression 路径。实现位置：

- `Wan21/pipeline/kv_bank.py`
  - 每个 block 入 bank 前，从 `viewmats` 计算 `translation_delta` 和 `rotation_delta_rad`。
  - 根据运动幅度计算 `motion_score`。
  - 用动态 `keep_ratio` 压缩 normal KV 和 PRoPE KV。
  - `[kv-bank]` 日志会打印每个 block 的 `keep_ratio` 和 `motion_score`。
- `Wan21/wan_inference.py`
  - 新增动态剪枝 CLI 参数。
- `Wan21/pipeline/causal_inference.py`
  - DMD / ODE 路径将动态剪枝参数传入 `KVBank`。
- `Wan21/pipeline/causal_diffusion_inference.py`
  - diffusion 路径将动态剪枝参数传入 `KVBank`。
- `Wan21/scripts/inference/run_infer_causal_camera.sh`
  - 新增对应环境变量，统一传给 `wan_inference.py`。

当前使用的连续策略是：

```text
motion_score = translation_delta / translation_scale + rotation_delta_rad / rotation_scale
keep_ratio = clamp(base_keep_ratio - motion_weight * motion_score, min_keep, max_keep)
```

默认参数：

```text
base_keep_ratio = 0.5
min_keep = 0.25
max_keep = 0.75
translation_scale = 1.0
rotation_scale = 0.35
motion_weight = 0.25
```

解释：相机运动越大，当前版本会更激进地压缩历史 block；相机运动较小时使用接近固定剪枝的 keep ratio。后续如果实验证明大运动更需要保留更多 token，可以将公式改为 `base + motion_weight * motion_score` 或加入 FOV overlap 正向项。

## 评估计划

### 控制变量

不同实验之间保持这些变量不变：

- prompt
- trajectory
- seed
- checkpoint
- resolution
- output frame 数
- denoising steps
- `local_attn_size`
- `num_frame_per_block`

### 建议实验组

```text
E0: 原始 minWM
E1: fixed sink
E2: periodic sink update
E3: WorldKV-style pose retrieval
E4: FOV retrieval
E5: hybrid retrieval
E6: pose retrieval + fixed pruning
E7: FOV/hybrid retrieval + dynamic pruning
```

### 一键实验矩阵脚本

当前已经提供 ViewBench small12 的矩阵 runner：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

默认设置：

- `MAX_PROMPTS=5`
- `NUM_OUTPUT_FRAMES=120`
- `SEEDS=0`
- `KV_BANK_MAX_BLOCKS=45`
- `RETRIEVAL_FRAMES=12`
- `RETRIEVAL_RECENT_FRAMES=32`
- `RETRIEVAL_FOV_SAMPLES=8192`

默认会依次跑：

```text
baseline
fixed_sink
periodic_sink
pose_compress_store
worldkv_fov_compress_store
```

快速排错建议先 dry-run：

```bash
DRY_RUN=1 \
CASES="baseline worldkv_fov_dynamic_compress_store" \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=20 \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

动态剪枝单独实验：

```bash
CASES="worldkv_fov_dynamic_compress_store" \
MAX_PROMPTS=5 \
NUM_OUTPUT_FRAMES=120 \
RUN_ROOT=./outputs/viewbench_worldkv_fov_dynamic_prune_120f \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

可调参数：

```bash
KV_COMPRESSION_DYNAMIC_MIN_KEEP=0.25
KV_COMPRESSION_DYNAMIC_MAX_KEEP=0.75
KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE=1.0
KV_COMPRESSION_DYNAMIC_ROTATION_SCALE=0.35
KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT=0.25
```

正式小规模比较：

```bash
SEEDS="0 1 2" \
MAX_PROMPTS=12 \
NUM_OUTPUT_FRAMES=180 \
RUN_ROOT=./outputs/viewbench_worldkv_fov_formal \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

每个 case 的输出目录形如：

```text
RUN_ROOT/seed_0/baseline/
RUN_ROOT/seed_0/worldkv_fov_compress_store/profile/
RUN_ROOT/seed_0/worldkv_fov_dynamic_compress_store/profile/
```

矩阵脚本会额外生成：

```text
RUN_ROOT/experiment_manifest.txt
RUN_ROOT/experiment_summary.tsv
```

### 推荐 trajectory

选择能考验长期记忆的轨迹：

- 向前移动后返回。
- 左右反复旋转。
- 绕圈后回到初始视角。
- 小幅震荡，FOV overlap 高。
- 大幅转向，FOV overlap 低。

### 诊断输出

即使暂时没有正式指标，也要保存：

- selected block ids over time
- 当前 block 与 bank block 的 FOV overlap heatmap
- 每个 block 的 keep ratio
- 每个 block 的 retrieved token 数
- 峰值显存
- 生成时间

## 风险清单

### Cache 一致性

minWM 同时有 normal KV 和 PRoPE KV。任何 sink、retrieval、compression 操作都必须一致处理两者。

### RoPE 位置修正

retrieved KV 是在历史时间位置编码出来的。如果把它插入到 recent 前面，可能需要类似 WorldKV 的 RoPE correction。

当前已经支持可选 normal KV RoPE rebasing：`RETRIEVAL_ROPE_CORRECTION=1`。PRoPE KV 仍沿用历史 block 存储时的位置编码，尚未做严格几何 rebasing。如果 retrieval 带来画面跳变或 camera-aware 几何不稳定，优先排查 PRoPE retrieved KV 的位置一致性。

### Token 数假设

compression 会改变 token 数。所有默认假设 `chunk_size * frame_seq_length` 的代码都需要检查。

### 显存与内存

初期优先使用 CPU KV bank。GPU bank 应该作为可选加速项，而不是默认路径。

### 实验归因

不要一开始同时加入 retrieval、compression、dynamic pruning。每次只加一个机制，避免结果无法解释。

## 里程碑 8：基于 KV bank 的 sink frame 更新策略

### 目标

在 fixed sink 和 periodic latest sink 之外，新增一组从历史 KV bank 中选择 sink block 的策略，用来测试“把哪些历史记忆放入 sink 槽”对长视频 loop closure 和 camera memory 的影响。

### 新增策略

新增 `SINK_STRATEGY`：

- `bank_random`：从满足条件的历史 KV bank block 中随机选择 1 个 block，随机过程由 `SINK_BANK_SEED` 固定，便于复现实验。
- `bank_uniform`：从满足条件的历史 KV bank block 中按顺序轮转选择，尽量覆盖不同历史位置。
- `bank_pose`：复用当前 WorldKV pose 距离，选择与当前 camera pose 最相似的历史 block。
- `bank_worldkv_fov`：复用当前 WorldKV FOV overlap 距离，选择与当前视锥重叠最高的历史 block。`bank_fov` 在代码里作为别名映射到这一策略。

这几种策略仍然使用 `SINK_SIZE` 控制 sink 槽大小，并使用 `SINK_UPDATE_INTERVAL` 控制多久更新一次。当前默认实验中 `SINK_SIZE=4`，而 minWM 每个 causal block 也是 4 latent frames，因此每次从 KV bank 中选 1 个历史 block 填入 sink 槽。

### 候选 block 过滤

bank sink 会过滤以下 block：

- 起始帧早于固定 sink 区域的 block，避免覆盖原始 sink 语义。
- 与当前 block 重叠或位于未来的 block。
- 距当前 block 太近的 block；这个间隔复用 `RETRIEVAL_RECENT_FRAMES`，例如 32 latent frames。
- 不包含当前分支的 block。DMD/ODE/CD DMD 路径使用 `main` 分支；diffusion 路径分别使用 `cond` 和 `uncond` 分支。

### 实现位置

- [Wan21/pipeline/sink_utils.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/sink_utils.py)：新增策略枚举、bank block 选择、bank KV 复制到 sink 槽、压缩 token 不足时清零尾部。
- [Wan21/pipeline/causal_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/causal_inference.py)：DMD/causal 路径在 clean context 更新后调用 `maybe_update_sink`。
- [Wan21/pipeline/causal_diffusion_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/causal_diffusion_inference.py)：diffusion 路径分别对 `cond/uncond` cache 调用 `maybe_update_sink`。
- [Wan21/wan_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/wan_inference.py)：新增 CLI 策略和 `--sink_bank_seed`。
- [Wan21/scripts/inference/run_infer_causal_camera.sh](/pool/hdd/home/hhhjyz/research/minWM/Wan21/scripts/inference/run_infer_causal_camera.sh)：新增 `SINK_BANK_SEED` 环境变量。
- [Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh](/pool/hdd/home/hhhjyz/research/minWM/Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh)：新增 `bank_random_sink`、`bank_uniform_sink`、`bank_pose_sink`、`bank_worldkv_fov_sink` 可选 case。

### 使用命令

单独测试一种 bank sink：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

RUN_ROOT=./outputs/viewbench_bank_sink_probe_120f \
CASES="baseline fixed_sink periodic_sink bank_random_sink bank_uniform_sink bank_pose_sink bank_worldkv_fov_sink" \
SEEDS="0" \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=120 \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=4 \
SINK_BANK_SEED=0 \
KV_BANK_MAX_BLOCKS=45 \
RETRIEVAL_RECENT_FRAMES=32 \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

如果只想快速看策略是否能正确触发：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

DRY_RUN=1 \
CASES="bank_random_sink bank_uniform_sink bank_pose_sink bank_worldkv_fov_sink" \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

### 当前风险

1. PRoPE KV 仍然是直接复制已编码后的历史 KV 到 sink 槽，没有重新按 sink 位置和当前 camera pose 进行严格重编码。这和之前 periodic sink 的风险一致。若 bank sink 画面出现视角错位、漂移或局部纹理突变，优先排查这里。
2. 如果开启 store-time compression，KV bank block 的 token 数可能小于 `SINK_SIZE * frame_seq_length`。当前实现会复制可用 token，并把剩余 sink token 清零，避免旧 sink 污染。但这意味着 compressed bank sink 与 full bank sink 的注意力容量不同，正式对比时建议先关闭 compression。
3. `bank_random` 是消融基线，不代表合理记忆策略；它主要用于判断“任意历史记忆进入 sink”是否会破坏生成。
4. `bank_pose` 和 `bank_worldkv_fov` 复用 retrieval metric，选择结果取决于 ViewBench pose 输入是否已经正确转换为 minWM 使用的 PRoPE camera pose。

## 里程碑 9：PRoPE KV 重编码策略

### 目标

在 retrieval 和 bank sink 两条路径中加入可选 PRoPE KV re-encoding，测试“历史 camera memory 是否应该被重编码到当前 camera pose”。

### 数学假设

minWM 的 PRoPE 对 K/V 使用相机投影矩阵的逆变换。简化地看：

```text
prope_kv_src = P_src_inv @ raw_kv
```

如果希望把已经存储的历史 PRoPE KV 重编码到当前 block 的相机 pose，可以近似做：

```text
raw_kv ~= P_src @ prope_kv_src
prope_kv_current = P_current_inv @ raw_kv
                 = P_current_inv @ P_src @ prope_kv_src
```

当前实现正是这个线性变换：`P_current_inv @ P_src`。

### 新增开关

新增：

```text
PROPE_REENCODE_MODE=none|current
```

- `none`：默认值，不改变已有实验行为。retrieval 和 sink 都继续使用存 bank 时已有的 PRoPE KV。
- `current`：retrieval payload 取出时、bank sink 写入 sink 槽前，尝试把历史 PRoPE KV 从历史 pose 重编码到当前 block pose。

### 实现位置

- [Wan21/pipeline/kv_bank.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/kv_bank.py)
  - 新增 `reencode_prope_kv_to_current`。
  - retrieval payload 中新增 `prope_reencode_mode`、`current_viewmats`、`current_Ks`。
  - `retrieval_events` 会记录 `prope_reencode_mode` 和 `prope_reencoded`。
- [Wan21/pipeline/sink_utils.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/sink_utils.py)
  - bank sink 复制 PRoPE KV 前支持重编码。
  - `[sink-update]` 日志会打印 `prope_reencode` 和 `prope_reencoded`。
- [Wan21/pipeline/causal_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/causal_inference.py)
  - DMD/causal 路径将当前 `vm_chunk/ks_chunk` 传入 retrieval 和 sink reencode。
- [Wan21/pipeline/causal_diffusion_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/pipeline/causal_diffusion_inference.py)
  - diffusion 路径对 `cond/uncond` 分支同步支持。
- [Wan21/wan_inference.py](/pool/hdd/home/hhhjyz/research/minWM/Wan21/wan_inference.py)
  - 新增 `--prope_reencode_mode`。
- [Wan21/scripts/inference/run_infer_causal_camera.sh](/pool/hdd/home/hhhjyz/research/minWM/Wan21/scripts/inference/run_infer_causal_camera.sh)
  - 新增 `PROPE_REENCODE_MODE`。
- [Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh](/pool/hdd/home/hhhjyz/research/minWM/Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh)
  - 矩阵实验会把 `PROPE_REENCODE_MODE` 写入 manifest 和 run meta。

### 使用命令

测试 retrieval PRoPE reencode：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

RUN_ROOT=./outputs/viewbench_prope_reencode_retrieval_120f \
CASES="worldkv_fov" \
SEEDS="0" \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=120 \
PROPE_REENCODE_MODE=current \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

测试 bank sink PRoPE reencode：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

RUN_ROOT=./outputs/viewbench_prope_reencode_sink_120f \
CASES="bank_pose_sink bank_worldkv_fov_sink" \
SEEDS="0" \
MAX_PROMPTS=1 \
NUM_OUTPUT_FRAMES=120 \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=4 \
PROPE_REENCODE_MODE=current \
bash Wan21/scripts/inference/run_viewbench_worldkv_fov_experiments.sh
```

建议正式对比时至少包含：

```text
worldkv_fov_compress_store + PROPE_REENCODE_MODE=none
worldkv_fov + PROPE_REENCODE_MODE=none
worldkv_fov + PROPE_REENCODE_MODE=current
bank_worldkv_fov_sink + PROPE_REENCODE_MODE=none
bank_worldkv_fov_sink + PROPE_REENCODE_MODE=current
```

### 当前限制和风险

1. 第一版只支持 `current` 目标 pose，不支持把历史 block 重编码到“虚拟历史时间轴 pose”。
2. 若 bank block 已经 store-time compression，当前实现会跳过 PRoPE reencode。原因是压缩后 token 与 frame 的规则排列可能被破坏，无法可靠 reshape 为 `(cameras, patches)`。
3. 如果 source block 和 current block 的 camera 数不同，也会跳过 reencode。
4. 这个策略不一定天然更好。PRoPE 的标准 cross-attention 解释通常是 query 使用当前 pose，key/value 使用自身 source pose；强行重编码到当前 pose 是一个实验假设，需要用 ViewBench loop closure 和 VBench-style 指标判断。
5. `prope_reencoded=0` 不一定是 bug，可能表示当前 block 缺 pose、source/destination camera 数不匹配，或 selected block 是 compressed block。

## 近期建议步骤

1. 跑通 minWM Wan21 Action2V quickstart。
2. 加一个最小日志开关，记录 cache index 和显存。
3. 在 camera DMD config 中尝试 `sink_size: 4`。
4. fixed sink 确认后，再实现 periodic sink update。
5. 在写 retrieval 前，单独设计 KV bank API。

## 维护日志

- 2026-07-18
  - 完成里程碑 8 第一版：新增基于 KV bank 的 sink frame 更新策略。
  - 支持 `bank_random`、`bank_uniform`、`bank_pose`、`bank_worldkv_fov`，并新增 `SINK_BANK_SEED` 控制随机消融可复现。
  - bank sink 会复用 KV bank 中的 normal KV / PRoPE KV，并在 compressed block token 不足时清零 sink 尾部，避免旧内容残留。
  - 已在文档中标注 PRoPE 直接复制不重编码的几何一致性风险。
  - 完成里程碑 9 第一版：新增 `PROPE_REENCODE_MODE=none|current`，在 retrieval 和 bank sink 路径中支持 PRoPE KV 重编码实验。

- 2026-07-17
  - 完成里程碑 7 第一版：在 store-time KV compression 中加入基于相机运动幅度的动态 keep ratio。
  - 动态剪枝公式：`keep_ratio = clamp(base - motion_weight * motion_score, min_keep, max_keep)`，其中 `motion_score` 由 block 内平移幅度与旋转角归一化得到。
  - 新增 `worldkv_fov_dynamic_compress_store` 实验 case；默认五组正式实验保持不变，需要动态剪枝时通过 `CASES` 显式指定。
  - 已完成 bash 语法检查、Python py_compile、动态 case dry-run 和 fake-cache 动态 keep ratio 验证。

- 2026-07-16
  - 完成里程碑 4 第一版：在 minWM causal attention 中加入 WorldKV 风格 `[sink | retrieved | recent]` retrieval window。
  - 完成里程碑 5 第一版：新增 `pose`、`worldkv_fov`、`hy_fov`、`hybrid`、`recent_only` 检索 metric。
  - 完成里程碑 6 第一版：新增 WorldKV-style anchor + novelty KV compression，支持 store-time 和 retrieval-time 两种模式。
  - 新增 `retrieval_rope_correction` / `RETRIEVAL_ROPE_CORRECTION`，支持 WorldKV-style normal KV time-axis RoPE rebasing。
  - 新增 `Wan21/pipeline/retrieval_utils.py`，集中实现 WorldKV 和 HY-WorldPlay 风格的检索距离。
  - 新增 `retrieval_events.csv/jsonl`，记录 block-level candidate/selected block、distance、retrieved token 数和检索耗时。
  - 新增 `run_viewbench_worldkv_fov_experiments.sh`，一键跑 baseline、sink、retrieval、FOV 和 compression 实验矩阵。
  - 已用 `ling` 环境完成 py_compile、bash 语法检查和 fake-cache 单元验证。
  - 完成里程碑 3：新增 chunk-level KV bank。
  - DMD/ODE 保存 main 分支，diffusion 保存 cond/uncond 分支。
  - 支持 CPU/GPU 存储、FIFO block 上限、内存预测警告和 timing 汇总。
  - 新增 `run_kv_bank_smoke_causal_camera.sh` 安全验证入口。

- 2026-07-14
  - 创建路线规划。
  - 确认 minWM Wan21 是更适合单卡 1.3B 实验的平台。
  - 确认 minWM causal attention 具备 KV cache、PRoPE KV cache、`local_attn_size` 和潜在的 `sink_size` 机制。
  - 将文档改写为中文版本，方便后续持续维护。
  - 完成里程碑 1：加入可开关的 cache-state 日志，覆盖 causal DMD 与 causal diffusion 推理路径。
# 历史实验计划

> 本文档保留早期 ViewBench 实验记录，仅用于结果追溯。当前推理入口已经统一为
> prompt + minWM action string；请以
> [`Wan21/README_STRING_CAMERA_EXPERIMENTS.md`](Wan21/README_STRING_CAMERA_EXPERIMENTS.md)
> 为准。文中 ViewBench pose、manifest 和 runner 路径已不再存在。
