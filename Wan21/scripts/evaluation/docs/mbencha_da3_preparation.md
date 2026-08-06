# MBench-A 的 DA3 artifacts 准备说明

## 目标与范围

MBench-A 的几何类指标需要 Depth Anything 3（DA3）结果。本项目的 14 个
`minwm_sink_rebase_retrieval_*_seed0` 模型中，每个模型有 547 个视频；DA3 只用于：

- `environment`：227 个；
- `object`：100 个；
- `causal`：100 个；
- `human`：120 个，不需要 DA3。

因此每个模型需要 427 份、全部 14 个模型需要 5978 份 DA3 artifact。

每份结果保存到：

```text
MBench-A/models/<model_id>/artifacts/<subset>/<sample_id>/<condition_id>/da3/
├── results.npz                 # depth、extrinsics、intrinsics
└── input_images/000000.png ... # DA3 实际处理后的图像
```

脚本将 DA3 返回的 OpenCV world-to-camera 外参转换为 MBench 几何公式采用的
camera-to-world 外参。图像使用 `processed_images`，从而与深度和内参分辨率一致。

## 环境安装

建议单独建立环境，避免再次影响 minWM 的 `transformers`/`huggingface-hub` 版本：

```bash
conda create -n da3 python=3.10 -y
conda activate da3
cd /home/jiangyize/research
git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
cd Depth-Anything-3
pip install -e .

# 本节点驱动支持 CUDA 12.8；不要保留 pip 自动选择的 CUDA 13 版本
pip install --force-reinstall torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install 'numpy<2'
pip install xformers==0.0.32.post2 \
  --index-url https://download.pytorch.org/whl/cu128
pip install 'httpx[socks]'
```

当前机器已经完成上述环境，路径为
`/home/jiangyize/software/miniconda3/envs/da3`。已验证版本为 PyTorch 2.8.0+cu128、
torchvision 0.23.0+cu128、xformers 0.0.32.post2 和 NumPy 1.26.4。

权重也已下载到 `outputs/mbencha_da3/hf_cache`。集群设置的 `HF_ENDPOINT=https://hf-mirror.com`
不兼容当前 `huggingface_hub` 的 metadata 请求，脚本会取消该变量并设置
`HF_HUB_DISABLE_XET=1`；后者使用较稳定、可续传的普通 HTTP 下载。

## 重新打包 14 个模型

生成视频的 timing CSV 已按 `prompt_index` 去重，但 dataset package 需要覆盖更新一次：

```bash
cd /home/jiangyize/research/minWM
/home/jiangyize/software/miniconda3/envs/minwm-fa/bin/python \
  Wan21/scripts/evaluation/mbencha_adapter.py package \
  --dataset-root /home/jiangyize/research/datasets/MBench-Data/MBench-A \
  --manifest outputs/mbencha_adapter/official_25s/manifest.jsonl \
  --run-root outputs/mbencha_sink_rebase_retrieval_ablation_25s_seed0 \
  --model-prefix minwm_sink_rebase_retrieval \
  --link-mode symlink
```

正确结果应当是 14 个模型各 547 行。

## 双卡生成 DA3 artifacts

物理 GPU 0、1：

```bash
cd /home/jiangyize/research/minWM
CUDA_DEVICES=0,1 \
DA3_PYTHON=/home/jiangyize/software/miniconda3/envs/da3/bin/python \
bash Wan21/scripts/evaluation/run_mbencha_da3.sh
```

每个 worker 内部只看到一张卡，因此都使用逻辑 `cuda:0`；launcher 会记录物理卡号。
任务采用稳定的 `task_index % world_size` 分片。脚本可重复运行：有效 artifact 自动跳过，
不完整 artifact 在启用 `OVERWRITE_INVALID=1` 时会先改名保留，再原子写入新结果。

launcher 默认每 15 秒在终端输出一次所有 GPU worker 的聚合进度，例如：

```text
[DA3 15:15:59] 80/200 (40.00%) | 新生成=49 跳过有效=30 失败=1 | active=2/2 | 4.99 samples/min | elapsed=10m ETA=24m
```

可以通过 `PROGRESS_INTERVAL` 调整刷新间隔：

```bash
PROGRESS_INTERVAL=30 CUDA_DEVICES=0,1,2,3 \
  bash Wan21/scripts/evaluation/run_mbencha_da3.sh
```

每个 rank 的机器可读状态保存在 `outputs/mbencha_da3/progress/`，完整模型输出仍写入
`outputs/mbencha_da3/logs/rank*.log`，避免多卡日志在终端交错。

一次真实 smoke test 的输入包含 397 帧：DA3 forward 为 28.1 秒，端到端为 51.5 秒，产物
约 291 MiB。按此样本粗略外推，5978 份产物约需 1.7 TiB，两卡约需 43 小时；不同视频内容
会使 PNG/NPZ 压缩率和耗时略有变化。建议在 `tmux` 中启动全量任务：

```bash
tmux new -s mbencha-da3
cd /home/jiangyize/research/minWM
CUDA_DEVICES=0,1 bash Wan21/scripts/evaluation/run_mbencha_da3.sh
# Ctrl-b d 可安全退出 tmux；之后用 tmux attach -t mbencha-da3 恢复
```

常用覆盖参数：

```bash
# 先只跑一个样本进行 smoke test
CUDA_VISIBLE_DEVICES=0 \
/home/jiangyize/software/miniconda3/envs/da3/bin/python \
  Wan21/scripts/evaluation/prepare_mbencha_da3.py \
  --dataset-root /home/jiangyize/research/datasets/MBench-Data/MBench-A \
  --model-glob 'minwm_sink_rebase_retrieval_sink_rebase_seed0' \
  --limit 1 --overwrite-invalid

# 使用更小显存占用（会改变 DA3 处理分辨率，整组实验必须保持一致）
PROCESS_RES=392 CUDA_DEVICES=0,1 bash Wan21/scripts/evaluation/run_mbencha_da3.sh
```

## 查看进度与校验

```bash
/home/jiangyize/software/miniconda3/envs/da3/bin/python \
  Wan21/scripts/evaluation/summarize_mbencha_da3.py \
  --dataset-root /home/jiangyize/research/datasets/MBench-Data/MBench-A \
  --output outputs/mbencha_da3/DA3_ARTIFACTS_PROGRESS.md
```

汇总器会实际打开 NPZ 并检查 key、shape、帧数、NaN/Inf 和正深度，而不只统计文件数。
worker 日志在 `outputs/mbencha_da3/logs/`，逐样本事件记录在
`outputs/mbencha_da3/events_rank*.jsonl`。

## 在两台机器上完全分离测评

全量 runner 支持通过 `RUN_GROUP` 将任务拆成互不管理、互不等待的进程：

- `RUN_GROUP=da3`：准备并校验 DA3，然后运行 4 项 DA3 依赖指标；
- `RUN_GROUP=vlm`：只运行 `state_progress` 和 `progress_correctness`，不要求 CUDA；
- `RUN_GROUP=non_vlm`：只运行其余 6 项本地非 VLM 指标，支持多卡；
- `RUN_GROUP=non_da3`：兼容模式，一起运行上述 8 项指标；
- `RUN_GROUP=all`：保留单机流水线模式，通常不用于双机实验。

两台机器必须看到同一个 `DATASET_ROOT`，因为 DA3 artifacts 保存在 MBench-A 的模型目录中。
最好也共享同一个 `OUTPUT_ROOT`，这样最终状态和指标结果集中在同一目录；如果文件系统不共享，需在
测评结束后再合并两个输出目录。

依赖 DA3、必须等待的 4 项是：

- `object_geometry_consistency`；
- `spatial_epipolar`；
- `spatial_reprojection`；
- `camera_interaction`。

MBench 侧可以使用模块入口查看当前指标：

```bash
cd /home/jiangyize/research/MBench
/home/jiangyize/software/miniconda3/envs/minwm-fa/bin/python -m mbench.cli list-metrics
```

正式命令应以当前 checkout 的 `python -m mbench.cli --help` 为准。runner 不会在 DA3
尚未齐全时启动几何指标，避免缺失 artifact 产生无意义失败记录。

项目已经提供包含 DA3、12 项官方指标和 VLM judge 的全量 runner：

```bash
cd /home/jiangyize/research/minWM
mkdir -p outputs/mbencha_full_evaluation
cp Wan21/scripts/evaluation/mbencha_vlm.env.example \
  outputs/mbencha_full_evaluation/vlm.env
chmod 600 outputs/mbencha_full_evaluation/vlm.env
# 编辑 vlm.env，填入真实 endpoint、key 和视觉模型名称

tmux new -s mbencha-full
# 机器 A：只运行 DA3 准备和 4 项 DA3 依赖指标
RUN_GROUP=da3 \
DA3_CUDA_DEVICES=0 \
EVAL_CUDA_DEVICE=0 \
PROCESS_RES=504 \
bash Wan21/scripts/evaluation/run_mbencha_full_evaluation.sh
```

远程 VLM 指标可以在单独的 CPU/API 节点运行：

```bash
cd /home/jiangyize/research/minWM
RUN_GROUP=vlm \
VLM_WORKERS=1 \
VLM_MAX_RETRIES=10 \
bash Wan21/scripts/evaluation/run_mbencha_full_evaluation.sh
```

本地非 VLM 指标在 GPU 节点运行：

```bash
cd /home/jiangyize/research/minWM
RUN_GROUP=non_vlm \
EVAL_CUDA_DEVICES=0,1,2,3 \
bash Wan21/scripts/evaluation/run_mbencha_full_evaluation.sh
```

`non_vlm` 模式采用任务级多卡并行：创建 4 个 worker，每个 worker 固定只看到一张物理卡，
并在该卡上串行运行分配到的指标。因此不会在单卡上同时加载两个本地评测模型。
`EVAL_CUDA_DEVICE` 仍可用于单卡兼容，但设置了 `EVAL_CUDA_DEVICES` 时以后者为准。

三组使用不同锁 `run.da3.lock`、`run.vlm.lock` 和 `run.non_vlm.lock`，因此可以在共享输出目录中同时运行。
API key 只通过环境传递，不会加入命令行或日志。`RUN_GROUP=da3` 不读取或检查 VLM 配置。
中断后执行完全相同的命令即可续跑；有效 DA3 artifact 和带 `.complete` 的指标都会跳过。日志位于
`outputs/mbencha_full_evaluation/logs/da3_stage.log` 及各指标的 `metrics/*/console.log`。
