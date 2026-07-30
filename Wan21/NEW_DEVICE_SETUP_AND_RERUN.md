# minWM 新设备环境配置与实验续跑

本文档用于把当前 Wan2.1 Action2V 字符串相机实验迁移到新设备，并继续运行
10s、15s、20s、30s 的 14 组实验。命令默认采用以下目录结构：

```text
<workspace>/
├── minWM/
├── ckpts/
│   ├── Wan2.1-T2V-1.3B/
│   └── Wan21/Action2V/dmd/model.pt
└── VBench/                       # 可选
```

不要把 checkpoint 或 `outputs/` 提交到 Git。它们需要单独下载或使用 `rsync`
传输。

## 1. 迁移前在旧设备提交并推送

当前字符串相机实验的基础提交为：

```text
3e30e30 Add string-camera long-video experiment suite
```

在旧设备执行：

```bash
cd /path/to/minWM
git log -1 --oneline
git push origin main
```

`git push` 只同步源码，不会同步被 `.gitignore` 排除的 checkpoint、视频和评测
结果。

## 2. 检查新设备资源

推荐资源：

- 单张完整 A100 80GB，或单个 `MIG 7g.80gb`；
- 至少 64GB 主存，推荐 128GB，因为 KV bank 默认存放在 CPU；
- 至少 100GB 可用磁盘，四种时长的 1680 个视频会继续增长；
- NVIDIA 驱动支持 CUDA 12.8；
- `git`、`ffmpeg`、`tmux`、`rsync`、C/C++ 编译器。

安装常用系统工具，例如：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs ffmpeg tmux rsync build-essential ninja-build
```

集群环境没有 `sudo` 时，应使用管理员提供的软件模块。

先检查 GPU：

```bash
nvidia-smi -L
nvidia-smi
```

后续安装 PyTorch 后，再使用下面的检查作为最终判据：

```bash
python - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA is unavailable"
p = torch.cuda.get_device_properties(0)
total_gb = p.total_memory / 1024**3
print("device:", p.name)
print("visible memory GiB:", total_gb)
assert total_gb >= 70, "Need a full A100 80GB or MIG 7g.80gb"
PY
```

物理卡名称包含 `A100 80GB` 并不代表当前进程拥有 80GB。若 PyTorch 输出
`MIG 1g.10gb` 和 `9.5 GiB`，WorldKV/PRoPE cache 会 OOM。同一张物理卡上的
多个 10GB MIG 也不能合并为一张大显存设备。

## 3. 获取代码

```bash
mkdir -p /path/to/workspace
cd /path/to/workspace
git clone git@github.com:hhhjyz/minWM.git
cd minWM
git checkout main
git pull --ff-only origin main
git merge-base --is-ancestor 3e30e30 HEAD
```

最后一条命令退出码为 0，表示代码包含字符串相机实验基础提交。检查当前版本：

```bash
git log -3 --oneline
git status -sb
```

## 4. 准备 checkpoint

### 4.1 从旧设备复制

在新设备执行，替换用户名、主机名和旧路径：

```bash
cd /path/to/workspace
mkdir -p ckpts
rsync -avP \
  user@old-host:/old/workspace/ckpts/Wan2.1-T2V-1.3B/ \
  ckpts/Wan2.1-T2V-1.3B/
rsync -avP \
  user@old-host:/old/workspace/ckpts/Wan21/Action2V/dmd/ \
  ckpts/Wan21/Action2V/dmd/
```

### 4.2 重新下载

```bash
python -m pip install -U "huggingface_hub[cli]"

cd /path/to/workspace
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir ckpts/Wan2.1-T2V-1.3B
hf download MIN-Lab/minWM \
  --local-dir ckpts \
  --include "Wan21/Action2V/dmd/*"
```

### 4.3 建立 Wan 基础模型软链接

```bash
cd /path/to/workspace/minWM
mkdir -p Wan21/wan_models
ln -sfn "$(realpath ../ckpts/Wan2.1-T2V-1.3B)" \
  Wan21/wan_models/Wan2.1-T2V-1.3B
```

验证关键文件：

```bash
test -s ../ckpts/Wan21/Action2V/dmd/model.pt
test -s Wan21/wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
test -s Wan21/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
test -s Wan21/wan_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors
du -sh ../ckpts/Wan21/Action2V/dmd ../ckpts/Wan2.1-T2V-1.3B
```

当前文件规模大约为 5.6GB 和 17GB。明显小于该规模通常表示下载不完整。

## 5. 配置推理环境

推荐优先使用方案 A，它可以最大程度复现旧设备的工作环境。

### 5.1 方案 A：使用 conda-pack 迁移现有 `ling` 环境

旧设备：

```bash
conda install -n base -y conda-pack
conda pack -n ling -o /tmp/minwm-ling.tar.gz
rsync -avP /tmp/minwm-ling.tar.gz user@new-host:/tmp/
```

新设备：

```bash
mkdir -p "$HOME/miniconda3/envs/ling"
tar -xzf /tmp/minwm-ling.tar.gz -C "$HOME/miniconda3/envs/ling"
source "$HOME/miniconda3/bin/activate" "$HOME/miniconda3/envs/ling"
conda-unpack
```

该方案要求新旧设备均为兼容的 Linux x86-64 环境，且新设备 NVIDIA 驱动能够
运行 CUDA 12.8 程序。

### 5.2 方案 B：从头创建环境

已验证的推理环境使用 Python 3.11 和以下 Torch 栈：

```text
torch          2.8.0+cu128
torchvision    0.23.0+cu128
torchaudio     2.8.0+cu128
triton         3.4.0
flash-attn     2.8.3
```

创建环境：

```bash
conda create -n ling python=3.11 -y
conda activate ling
python -m pip install -U pip setuptools wheel packaging ninja

python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

仓库 `requirements.txt` 中包含另一组 Torch/CUDA 精确版本，不能直接覆盖上述
环境。过滤掉 Torch、Triton、TorchAO 和 NVIDIA wheel 后安装其余依赖：

```bash
grep -Ev \
  '^(torch|torchvision|torchaudio|torchao|triton|nvidia-[^=]+)==' \
  requirements.txt > /tmp/minwm-requirements-no-torch.txt
python -m pip install -r /tmp/minwm-requirements-no-torch.txt
```

安装 FlashAttention。优先使用与 Python 3.11、Torch 2.8、CUDA 12 匹配的预编译
wheel：

```bash
python -m pip install /path/to/flash_attn-2.8.3+cu12torch2.8*-cp311-linux_x86_64.whl
```

没有 wheel 时可尝试源码构建：

```bash
MAX_JOBS=8 python -m pip install flash-attn==2.8.3 --no-build-isolation
```

源码构建需要 CUDA toolkit、`nvcc` 和足够的主存。不要安装
`torchao==0.15.0`，它会在 Torch 2.8 下输出 C++ 扩展不兼容警告，而当前实验
不依赖 TorchAO。

### 5.3 配置 Python 路径并验证

```bash
cd /path/to/workspace/minWM
conda activate ling
export PYTHONPATH="$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"

python - <<'PY'
import av
import cv2
import flash_attn
import numpy
import omegaconf
import torch
import torchvision

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA runtime:", torch.version.cuda)
print("flash-attn:", flash_attn.__version__)
p = torch.cuda.get_device_properties(0)
print("GPU:", p.name, p.total_memory / 1024**3, "GiB")
assert torch.__version__.startswith("2.8.0")
assert p.total_memory / 1024**3 >= 70
PY
```

`pip check` 可能报告 `decord 0.6.0 is not supported on this platform`，但只要
`python -c "import decord"` 成功，这通常是旧 wheel 元数据问题。

## 6. 可选：迁移已有实验输出并断点续跑

当前 10s 实验已经生成的文件不在 Git 中。需要保留进度时，在新设备执行：

```bash
cd /path/to/workspace/minWM
mkdir -p outputs
rsync -avP --partial \
  user@old-host:/old/workspace/minWM/outputs/string_loop_10s_all_cases_seed0/ \
  outputs/string_loop_10s_all_cases_seed0/
```

也可以一并迁移其他时长：

```bash
rsync -avP --partial \
  'user@old-host:/old/workspace/minWM/outputs/string_loop_*_all_cases_seed0/' \
  outputs/
```

迁移后不要手工改视频文件名。第一次续跑会重新写入新设备上的绝对
`output_path`，状态为 `skipped_exists` 的视频会被视为已完成并参与评测。

## 7. 新设备 smoke test

先只运行一个 prompt 的 baseline 和 WorldKV FOV，确认完整推理路径均能工作：

```bash
cd /path/to/workspace/minWM
conda activate ling
export PYTHONPATH="$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
RUN_ROOT=./outputs/new_device_smoke_10s \
DATA_PATH=Wan21/prompts/demos_loop_closure/prompts.txt \
TRAJECTORY_PATH=Wan21/prompts/demos_loop_closure/trajectories_10s.txt \
NUM_OUTPUT_FRAMES=40 \
MAX_PROMPTS=1 \
SEEDS=0 \
CASES="baseline worldkv_fov" \
SINK_SIZE=4 \
KV_BANK_DEVICE=cpu \
KV_BANK_MAX_BLOCKS=10 \
RETRIEVAL_FRAMES=8 \
RETRIEVAL_RECENT_FRAMES=8 \
EVAL_STYLE_ENABLE=0 \
bash Wan21/scripts/inference/run_string_camera_experiments.sh
```

验证结果：

```bash
find outputs/new_device_smoke_10s -name '*.mp4' | wc -l
grep -R "Traceback\|OutOfMemoryError" outputs/new_device_smoke_10s || true
```

应得到 2 个视频且没有 traceback。

## 8. 运行或续跑全部时长

统一 watchdog 会按照以下顺序运行：

```text
10s: 40 latent frames
15s: 60 latent frames
20s: 80 latent frames
30s: 120 latent frames
```

每种时长包括 30 条 prompt 和 14 个 case，共 420 个视频。watchdog 会：

- 每分钟检查 PyTorch 实际可见显存；
- 显存低于 70GB 时等待，不会反复触发 OOM；
- 使用 `SKIP_COMPLETED=1` 保留已有视频；
- 每个时长达到 420/420 后才进入下一时长；
- 使用文件锁防止启动重复队列。

watchdog 当前统一使用 `KV_BANK_MAX_BLOCKS=10`，以复现当前队列的配置。若要改变
KV bank 容量，应先完成一组独立的小规模显存测试，并为新配置使用新的输出目录，
避免与已有结果混合。

在 `tmux` 中启动：

```bash
cd /path/to/workspace/minWM
conda activate ling

tmux new-session -d -s minwm_loop_watchdog \
  "cd '$PWD' && env \
   PYTHON_BIN='$CONDA_PREFIX/bin/python' \
   GPU_MIN_MEMORY_GB=40 \
   POLL_SECONDS=60 \
   RETRY_SECONDS=300 \
   bash Wan21/scripts/inference/run_string_loop_all_durations_watchdog.sh"
```

查看状态：

```bash
tmux list-sessions
tmux attach -t minwm_loop_watchdog
tail -f outputs/string_loop_all_durations_watchdog.log
tail -f outputs/string_loop_10s_all_cases_seed0/run_all_cases.log
```

离开 `tmux` 而不中止任务：按 `Ctrl+B`，再按 `D`。

## 9. 验证所有生成任务完成

```bash
cases='baseline fixed_sink periodic_sink bank_random_sink bank_uniform_sink bank_pose_sink bank_worldkv_fov_sink pose worldkv_fov hy_fov hybrid pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store'

for label in 10s 15s 20s 30s; do
  root="outputs/string_loop_${label}_all_cases_seed0/seed_0"
  total=0
  for case_name in $cases; do
    count=$(find "$root/$case_name" -maxdepth 1 -name '*.mp4' | wc -l)
    printf '%-4s %-45s %2d/30\n' "$label" "$case_name" "$count"
    total=$((total + count))
  done
  echo "$label total: $total/420"
done
```

完成条件是四种时长均为 `420/420`，总计 1680 个视频。还应检查：

```bash
grep -R "Traceback\|OutOfMemoryError" \
  outputs/string_loop_*_all_cases_seed0/*/profile/inference.log || true
```

历史失败日志可能仍存在，因此最终应以每个 case 的视频数量、最新
`experiment_summary.tsv` 和 watchdog 的 `verified_complete` 为准。

## 10. 可选：配置官方 VBench 环境

VBench 建议使用独立环境，不要修改 `ling` 推理环境：

```bash
cd /path/to/workspace
git clone https://github.com/Vchitect/VBench.git

conda create -n vbench python=3.10 -y
conda activate vbench
python -m pip install -U pip setuptools wheel
python -m pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e ./VBench
```

当前默认评测的七个基础质量维度不需要 Detectron2：

```text
subject_consistency
background_consistency
temporal_flickering
motion_smoothness
dynamic_degree
aesthetic_quality
imaging_quality
```

视频全部生成后，在 `ling` 环境调用批量适配器，实际 VBench 计算由独立环境
完成：

```bash
cd /path/to/workspace/minWM
conda activate ling

python Wan21/scripts/evaluation/run_official_vbench_batch.py \
  --root outputs/string_loop_10s_all_cases_seed0 \
  --root outputs/string_loop_15s_all_cases_seed0 \
  --root outputs/string_loop_20s_all_cases_seed0 \
  --root outputs/string_loop_30s_all_cases_seed0 \
  --minwm-root "$PWD" \
  --vbench-root ../VBench \
  --vbench-python "$HOME/miniconda3/envs/vbench/bin/python" \
  --vbench-cache-dir ../VBench/pretrained/cache \
  --load-ckpt-from-local \
  --output-dir outputs/official_vbench_all_durations
```

安装了 `latexmk`、`pdflatex`、`xelatex`、`lualatex` 或 `tectonic` 时，可增加
`--compile-pdf`。

## 11. 常见问题

### 11.1 `torch.OutOfMemoryError` 显示总显存只有 9.5GiB

当前进程拿到的是 `MIG 1g.10gb`。不要降低实验参数后混入正式对比结果，应申请
完整 A100 80GB 或 `MIG 7g.80gb`。

### 11.2 多个 10GB MIG 能否合并

不能。同一物理 A100 的多个 MIG 不共享显存，当前 NCCL 还会报告
`Duplicate GPU detected`。不同物理 GPU 才适合 sequence parallel。

### 11.3 `EADDRINUSE`

统一实验 runner 会自动选择空闲 master port。不要同时启动多个相同输出目录的
runner；watchdog 的文件锁也会阻止重复队列。

### 11.4 Detectron2 构建时找不到 Torch

仅在确实需要依赖 Detectron2 的 VBench 语义维度时安装，并关闭构建隔离：

```bash
python -m pip install --no-build-isolation \
  'detectron2@git+https://github.com/facebookresearch/detectron2.git'
```

Detectron2 对 CUDA/Torch 版本敏感，不属于当前七个基础质量维度的必要依赖。

### 11.5 watchdog 一直显示 `waiting_for_gpu`

执行 GPU 检查脚本确认 PyTorch 可见显存。watchdog 默认要求至少 70GB，这是为了
保持所有 case 的参数一致，而不是故障。资源满足后无需手工重启，它会自动继续。
