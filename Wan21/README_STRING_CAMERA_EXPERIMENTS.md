# minWM 字符串相机轨迹实验指南

本文档描述当前 Wan2.1 Action2V 推理流程。输入只包含：

- prompt 文本；
- minWM 标准字符串 camera action，例如 `w*19` 或 `w*40,j*40,w*39`。

仓库不再依赖 ViewBench pose、参考视频或 `.npz` 相机文件。官方 VBench 仍可用于评估生成视频的无参考质量指标。

在新设备上重新配置环境、复制 checkpoint、迁移已有输出和断点续跑，请参阅
[`NEW_DEVICE_SETUP_AND_RERUN.md`](NEW_DEVICE_SETUP_AND_RERUN.md)。

## 1. 环境与模型

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

export PYTHONPATH="$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"
```

需要以下文件：

```text
../ckpts/Wan21/Action2V/dmd/model.pt
Wan21/wan_models/Wan2.1-T2V-1.3B/
```

如果基础模型位于公共 checkpoint 目录，可以建立软链接：

```bash
mkdir -p Wan21/wan_models
ln -s /pool/hdd/home/hhhjyz/research/ckpts/Wan2.1-T2V-1.3B \
  Wan21/wan_models/Wan2.1-T2V-1.3B
```

## 2. Camera action 格式

每个 action 段的格式为 `<动作>*<重复次数>`，多个段使用逗号连接。

| Action | 相机运动 |
|---|---|
| `w` | 向前移动 |
| `s` | 向后移动 |
| `a` | 向左移动 |
| `d` | 向右移动 |
| `u` | 向上移动 |
| `dn` | 向下移动 |
| `j` | 向左转动 yaw |
| `l` | 向右转动 yaw |
| `i` | 向上转动 pitch |
| `k` | 向下转动 pitch |

示例：

```text
w*19
a*8,d*11
w*40,j*40,w*39
```

轨迹始终包含一个初始 identity pose，因此：

```text
pose 数量 = 1 + 所有 action 重复次数之和
```

`NUM_OUTPUT_FRAMES=120` 时，action 次数之和至少为 119。DMD camera 模型的 `num_frame_per_block=4`，所以 `NUM_OUTPUT_FRAMES` 必须是 4 的倍数。

字符串轨迹使用的归一化内参为：

```text
fx=0.5050505, fy=0.89786756, cx=0.5, cy=0.5
```

## 3. 重跑原生 demo

原生 demo 使用一行 prompt 对应一行 camera action：

```text
Wan21/prompts/demos.txt
Wan21/prompts/trajectories.txt
```

运行全部 demo：

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY_PATH=Wan21/prompts/trajectories.txt \
NUM_OUTPUT_FRAMES=20 \
MAX_PROMPTS=-1 \
SEED=0 \
OUTPUT_FOLDER=./outputs/string_demo_seed0 \
bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

只运行前 5 个：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PROMPTS=5 \
SEED=0 \
OUTPUT_FOLDER=./outputs/string_demo_first5_seed0 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

只测试一个 prompt 和一条指定轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY_PATH="" \
TRAJECTORY="j*19" \
NUM_OUTPUT_FRAMES=20 \
MAX_PROMPTS=1 \
PROMPT_START=0 \
SEED=0 \
OUTPUT_FOLDER=./outputs/string_demo_yaw_left \
bash Wan21/scripts/inference/run_infer_causal_camera.sh
```

当 `TRAJECTORY_PATH` 非空时，它优先于 `TRAJECTORY`。action 文件中的非空行数必须覆盖所选 prompt 的原始索引。

## 4. 120 帧长视频 baseline

原始 `trajectories.txt` 只有 20 个 pose，长视频实验应提供一条足够长的全局轨迹，或者新建逐行对齐的长轨迹文件。

下面让 5 个 prompt 使用同一条 120-pose 轨迹：

```bash
CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY_PATH="" \
TRAJECTORY="w*40,j*40,w*39" \
NUM_OUTPUT_FRAMES=120 \
MAX_PROMPTS=5 \
SEED=0 \
SINK_STRATEGY=none \
KV_BANK_ENABLE=0 \
RETRIEVAL_ENABLE=0 \
KV_COMPRESSION_ENABLE=0 \
OUTPUT_FOLDER=./outputs/string_long120_baseline_seed0 \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

推理后主要产物为：

```text
OUTPUT_FOLDER/*.mp4
OUTPUT_FOLDER/inference_times.csv
OUTPUT_FOLDER/inference_times.json
OUTPUT_FOLDER/retrieval_events.csv
OUTPUT_FOLDER/profile/inference.log
OUTPUT_FOLDER/profile/profile_summary.json
```

### 4.1 预构造的 30 条 loop-closure 轨迹

现有 30 条 demo prompt 已分别构造约 10、15、20、30 秒的闭环 action 文件：

```text
Wan21/prompts/demos_loop_closure/trajectories_10s.txt   # 40 poses
Wan21/prompts/demos_loop_closure/trajectories_15s.txt   # 60 poses
Wan21/prompts/demos_loop_closure/trajectories_20s.txt   # 80 poses
Wan21/prompts/demos_loop_closure/trajectories_30s.txt   # 120 poses
```

详细的构造方法、闭环误差和四种时长运行命令见
[`prompts/demos_loop_closure/README.md`](prompts/demos_loop_closure/README.md)。

## 5. 一次运行多组实验

统一实验入口：

```text
Wan21/scripts/inference/run_string_camera_experiments.sh
```

在 A100 80GB 上运行 5 个 prompt、120 帧、3 个 seed 和默认 6 个 case：

```bash
CUDA_VISIBLE_DEVICES=0 \
RUN_ROOT=./outputs/string_camera_all_cases_120f \
DATA_PATH=Wan21/prompts/demos.txt \
TRAJECTORY_PATH="" \
TRAJECTORY="w*40,j*40,w*39" \
NUM_OUTPUT_FRAMES=120 \
MAX_PROMPTS=5 \
SEEDS="0 1 2" \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=8 \
KV_BANK_DEVICE=cpu \
KV_BANK_MAX_BLOCKS=45 \
RETRIEVAL_FRAMES=12 \
RETRIEVAL_RECENT_FRAMES=32 \
KV_COMPRESSION_KEEP_RATIO=0.5 \
bash Wan21/scripts/inference/run_string_camera_experiments.sh
```

默认 case：

| Case | 说明 |
|---|---|
| `baseline` | 原始 local KV，不启用额外策略 |
| `fixed_sink` | 固定保留初始 sink frames |
| `periodic_sink` | 每隔 N 个 block 用最新帧更新 sink |
| `pose_compress_store` | pose 检索，存入 KV bank 时压缩 |
| `worldkv_fov_compress_store` | fixed sink + WorldKV FOV 检索，存入时压缩 |
| `worldkv_fov_dynamic_compress_store` | fixed sink + FOV 检索，根据相机运动动态调整 keep ratio |

可以通过 `CASES` 指定任意子集：

```bash
CASES="baseline fixed_sink periodic_sink" \
SEEDS="0" \
MAX_PROMPTS=1 \
bash Wan21/scripts/inference/run_string_camera_experiments.sh
```

runner 支持的全部 case：

```text
baseline
fixed_sink
periodic_sink
bank_random_sink
bank_uniform_sink
bank_pose_sink
bank_worldkv_fov_sink
pose
worldkv_fov
hy_fov
hybrid
pose_compress_store
worldkv_fov_compress_store
worldkv_fov_dynamic_compress_store
```

其中 `worldkv_fov`、`worldkv_fov_compress_store` 和
`worldkv_fov_dynamic_compress_store` 都同时启用 fixed sink，大小由
`SINK_SIZE` 控制。`bank_worldkv_fov_sink` 则是使用 FOV 检索结果动态替换
sink，二者不是同一种 sink 策略。

在真正运行前查看展开后的命令：

```bash
DRY_RUN=1 \
CASES="baseline pose worldkv_fov" \
bash Wan21/scripts/inference/run_string_camera_experiments.sh
```

## 6. 自动评测

### 6.1 轻量无参考指标

实验 runner 默认启用轻量 VBench-style proxy 指标：

```bash
EVAL_STYLE_ENABLE=1
EVAL_STYLE_MAX_FRAMES=96
EVAL_STYLE_RESIZE_WIDTH=256
```

结果保存在：

```text
<case>/eval/vbench_style_metrics.csv
<case>/eval/vbench_style_metrics.json
```

这些是内部 proxy 指标，不是官方 VBench 分数。

### 6.2 官方 VBench

字符串 prompt 可以直接交给官方 VBench 的 `custom_input` 模式，不需要使用 VBench 官方 prompt。

```bash
EVAL_OFFICIAL_VBENCH_ENABLE=1 \
OFFICIAL_VBENCH_ROOT=../Forcing-KV/evaluation/VBench \
OFFICIAL_VBENCH_PYTHON=/home/hhhjyz/miniconda3/envs/vbench/bin/python \
OFFICIAL_VBENCH_DIMENSIONS="subject_consistency background_consistency temporal_flickering motion_smoothness aesthetic_quality imaging_quality" \
bash Wan21/scripts/inference/run_string_camera_experiments.sh
```

官方结果保存在：

```text
<case>/eval/official_vbench_metrics.csv
<case>/eval/official_vbench_metrics.json
```

`dynamic_degree` 依赖 RAFT 权重，确认相应权重可用后再加入 `OFFICIAL_VBENCH_DIMENSIONS`。

由于不再使用配对参考视频，PSNR、SSIM 和 LPIPS 不属于当前默认评测。这些指标需要逐帧对齐的真实参考视频，不能用来评价任意 prompt 生成结果。

### 6.3 长时长实验 watchdog

当 GPU 资源可能在完整 A100 和 MIG 分片之间切换时，可以使用 watchdog。
它会检查 PyTorch 实际可见的显存，默认低于 70GB 时等待，不会在 10GB MIG
实例上反复触发 OOM。资源恢复后会依靠 `SKIP_COMPLETED=1` 自动续跑
10s、15s、20s 和 30s：

```bash
conda activate ling

PYTHON_BIN=python \
GPU_MIN_MEMORY_GB=70 \
POLL_SECONDS=60 \
bash Wan21/scripts/inference/run_string_loop_all_durations_watchdog.sh
```

状态日志位于：

```text
outputs/string_loop_all_durations_watchdog.log
```

脚本使用文件锁避免启动多个重复队列。每种时长只有在全部 14 个 case 都达到
`MAX_PROMPTS` 个视频后，才会进入下一种时长。

## 7. 参数总表

### 7.1 基础推理

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `CONFIG_PATH` | `Wan21/configs/causal_forcing_dmd_camera.yaml` | 推理 YAML |
| `CHECKPOINT_PATH` | `../ckpts/Wan21/Action2V/dmd/model.pt` | DMD checkpoint |
| `DATA_PATH` | `Wan21/prompts/demos.txt` | prompt 文件 |
| `TRAJECTORY` | `w*19` | 所有 prompt 共用的字符串轨迹 |
| `TRAJECTORY_PATH` | 空；smoke/runner 有各自默认值 | 一行 prompt 对应一行 action |
| `OUTPUT_FOLDER` | `output/causal_camera` | 单次推理输出目录 |
| `NUM_OUTPUT_FRAMES` | `20` | 模型时间帧数，必须是 4 的倍数 |
| `MAX_PROMPTS` | `-1` | 最多运行多少条，`-1` 表示全部 |
| `PROMPT_START` | `0` | 从原 prompt 文件的哪个索引开始 |
| `SEED` | `0` | 单次推理随机种子 |
| `SP_SIZE` | `1` | sequence parallel 大小 |
| `CUDA_VISIBLE_DEVICES` | 未设置 | 可见 GPU |
| `MASTER_ADDR` | `localhost` | torchrun rendezvous 地址 |
| `MASTER_PORT` | `29622` | torchrun 端口 |

### 7.2 Sink 与 KV bank

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SINK_STRATEGY` | `none` | `none/fixed/periodic/bank_random/bank_uniform/bank_pose/bank_worldkv_fov` |
| `SINK_SIZE` | `0` | sink 大小，单位为模型时间帧 |
| `SINK_UPDATE_INTERVAL` | `0` | 更新间隔，单位为生成 block |
| `SINK_BANK_SEED` | `0` | random bank sink 的随机种子 |
| `KV_BANK_ENABLE` | `0` | 是否保存额外 KV bank |
| `KV_BANK_DEVICE` | `cpu` | `cpu` 或 `cuda` |
| `KV_BANK_MAX_BLOCKS` | `0` | 最大 block 数，`0` 表示不限制 |
| `KV_BANK_LOG_INTERVAL` | `1` | 每 N 个 block 输出 bank 日志 |
| `KV_BANK_WARN_MEMORY_GB` | `16` | 预计 bank 内存超过该值时警告 |

### 7.3 Retrieval

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `RETRIEVAL_ENABLE` | `0` | 启用历史 KV 检索 |
| `RETRIEVAL_METRIC` | `pose` | `recent_only/pose/worldkv_fov/hy_fov/hybrid` |
| `RETRIEVAL_FRAMES` | `0` | 每次目标检索帧数 |
| `RETRIEVAL_RECENT_FRAMES` | `0` | 排除最近多少帧，避免只检索邻近 block |
| `RETRIEVAL_FOV_SAMPLES` | `8192` | FOV 几何探针数量 |
| `RETRIEVAL_FOV_RADIUS` | `8.0` | FOV 探针半径 |
| `RETRIEVAL_FOV_H_DEG` | `60.0` | 水平 FOV |
| `RETRIEVAL_FOV_V_DEG` | `35.0` | 垂直 FOV |
| `RETRIEVAL_HYBRID_FOV_WEIGHT` | `0.5` | hybrid 中 FOV 距离权重 |
| `RETRIEVAL_ROPE_CORRECTION` | `0` | 对 retrieved normal KV 进行时间 RoPE 修正 |
| `PROPE_REENCODE_MODE` | `none` | `none/current`，是否按当前相机重编码 PRoPE KV |

### 7.4 KV compression

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `KV_COMPRESSION_ENABLE` | `0` | 启用 KV token 压缩 |
| `KV_COMPRESSION_KEEP_RATIO` | `0.5` | 非 anchor 帧保留比例 |
| `KV_COMPRESSION_ANCHOR_ROTATE` | `0` | runtime compression 时轮换 anchor |
| `KV_COMPRESSION_AT_STORE` | `0` | 存入 bank 时只压缩一次 |
| `KV_COMPRESSION_POOLED` | `0` | 在非 anchor 帧间共享 keep budget |
| `KV_COMPRESSION_DYNAMIC_ENABLE` | `0` | 根据相机运动调整 keep ratio |
| `KV_COMPRESSION_DYNAMIC_MIN_KEEP` | `0.25` | 动态 keep ratio 下限 |
| `KV_COMPRESSION_DYNAMIC_MAX_KEEP` | `0.75` | 动态 keep ratio 上限 |
| `KV_COMPRESSION_DYNAMIC_TRANSLATION_SCALE` | `1.0` | 平移归一化尺度 |
| `KV_COMPRESSION_DYNAMIC_ROTATION_SCALE` | `0.35` | 旋转归一化尺度，单位 rad |
| `KV_COMPRESSION_DYNAMIC_MOTION_WEIGHT` | `0.25` | motion score 对 keep ratio 的影响权重 |

### 7.5 实验 runner 与日志

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `RUN_ROOT` | 带时间戳的目录 | 实验矩阵根目录 |
| `CASES` | 默认 6 个 case | 空格分隔的 case 列表 |
| `SEEDS` | `0` | 空格分隔的 seed 列表 |
| `LOG_CACHE_STATE` | `0` | 输出详细 cache 状态 |
| `LOG_CACHE_INTERVAL` | `1` | cache 日志间隔 |
| `EVAL_STYLE_ENABLE` | `1` | 自动运行轻量 proxy 评测 |
| `EVAL_OFFICIAL_VBENCH_ENABLE` | `0` | 自动运行官方 VBench |
| `OFFICIAL_VBENCH_ROOT` | `../Forcing-KV/evaluation/VBench` | 官方 VBench 仓库 |
| `OFFICIAL_VBENCH_PYTHON` | vbench 环境 Python | VBench Python 解释器 |
| `OFFICIAL_VBENCH_DIMENSIONS` | 六个质量维度 | 空格分隔的官方指标 |
| `SKIP_COMPLETED` | `0` | 检测完整 `inference_times.csv` 后跳过生成 |
| `CONTINUE_ON_ERROR` | `0` | 单个 case 失败后是否继续 |
| `DRY_RUN` | `0` | 只打印命令，不运行 |
| `PYTHON_BIN` | `python` | runner 和轻量评测使用的 Python |

## 8. 推荐实验顺序

1. 使用 `NUM_OUTPUT_FRAMES=20`、单 prompt 验证 `w/s/a/d/j/l/i/k` 的方向。
2. 使用 `NUM_OUTPUT_FRAMES=120` 跑 baseline，人工确认相机运动持续生效。
3. 固定 prompt、trajectory 和 seed，运行 sink 对比。
4. 再运行 pose/FOV retrieval 与 compression 对比。
5. 人工确认轨迹正常后，启用官方 VBench。
6. 正式结果使用 `SEEDS="0 1 2"`，报告均值和标准差。

不要在不同 case 之间改变 prompt、action、checkpoint、帧数或 seed。否则质量差异不能归因于 sink、retrieval 或 compression 策略。
