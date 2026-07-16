# minWM + WorldKV + FOV 检索路线规划

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

### 小样本 smoke test

完整 `demos.txt` 当前有 30 条 prompt，逐条全部推理会很慢。为了研究 sink frame 和 KV cache 对长视频生成的影响，默认使用 `tartanair_long5_120` 小数据集：每次跑 5 个视频，每个视频默认 120 帧。

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

### 带耗时 profiling 的 smoke test

如果需要在推理结束后自动保存每个视频的耗时，使用 profiling 入口：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

MAX_PROMPTS=5 \
NUM_OUTPUT_FRAMES=80 \
OUTPUT_FOLDER=./outputs/profile_tartanair_long5_120 \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

生成内容会保存在：

```text
./outputs/profile_tartanair_long5_120/profile/
```

主要文件：

- `inference.log`：完整推理日志。
- `inference_times.csv`：每个视频的耗时明细。
- `profile_summary.json` / `profile_summary.txt`：平均每视频耗时、平均生成耗时、写视频耗时、wall time 和估计 FPS。
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
NUM_OUTPUT_FRAMES=120 \
OUTPUT_FOLDER=./outputs/smoke_start0_n2_120 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```

参数含义：

- `DATA_PATH=Wan21/prompts/tartanair_long5_120/prompts.txt`：默认使用 5 条 TartanAir 风格长视频 prompt。
- `TRAJECTORY_PATH=Wan21/prompts/tartanair_long5_120/trajectories.txt`：默认使用与 prompt 对齐的长 camera trajectory。
- `MAX_PROMPTS=5`：默认跑 5 条 prompt。
- `PROMPT_START=0`：从第 0 条 prompt 开始；例如设为 `5` 就跑第 5 条开始的样本。
- `NUM_OUTPUT_FRAMES=120`：默认生成更长的视频，方便观察长时一致性。causal camera 默认 `num_frame_per_block=4`，建议使用 4 的倍数，例如 80、120、160。
- `TRAJECTORY_PATH` 仍然使用完整 trajectory 文件；代码会按原始 prompt 下标自动取对应行。

`tartanair_long5` 的构造脚本和说明：

- `minWM/Wan21/scripts/data_preprocessing/build_tartanair_long5.py`
- `minWM/Wan21/prompts/tartanair_long5_120/README.md`

如果只想临时绕过代码参数，也可以手动切小文件：

```bash
head -n 1 Wan21/prompts/demos.txt > Wan21/prompts/demos_1.txt
head -n 1 Wan21/prompts/trajectories.txt > Wan21/prompts/trajectories_1.txt
```

### 需要记录

- 峰值显存。
- 每个 prompt 的推理时间。
- 输出视频质量。
- camera trajectory 是否明显生效。
- 是否有依赖、checkpoint 路径或权重加载问题。

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
NUM_OUTPUT_FRAMES=80 \
SINK_SIZE=4 \
SINK_UPDATE_INTERVAL=4 \
RUN_PREFIX=sink_ablation_tartanair120 \
bash Wan21/scripts/inference/run_sink_ablation_causal_camera.sh
```

输出目录示例：

```text
outputs/sink_ablation_tartanair120_baseline_0_1_120/
outputs/sink_ablation_tartanair120_fixed_sink4_0_1_120/
outputs/sink_ablation_tartanair120_periodic_sink4_int4_0_1_120/
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

## 里程碑 3：在 minWM 中建立 KV bank

### 目标

先只保存历史 clean-block KV，不进行 retrieval。这样可以单独验证 bank 的存储逻辑和显存/内存开销。

### 存储时机

每个 block 生成后：

```text
denoise 当前 block
把输出写入 output
运行 clean context pass 更新 KV
抽取当前 block 的 clean KV
append 到 KV bank
```

### bank 内容

每一层保存：

- normal K
- normal V
- PRoPE K
- PRoPE V

每个 block 额外保存元信息：

- block id
- frame start/end
- viewmats/Ks slice
- 可选的 pose summary，供检索使用

### 设备策略

优先使用 CPU bank，保护 A100 40G 显存：

```text
bank tensor 存在 CPU
只有被检索到时才搬到 GPU
```

### 成功标准

- 每生成一个 block，bank 长度增加 1。
- retrieval 关闭时，bank 存储不影响生成结果。
- CPU 内存增长可预测。

## 里程碑 4：加入 WorldKV 风格 retrieval window

### 目标

使用 KV bank 组成 WorldKV 风格 attention window：

```text
[sink | retrieved | recent]
```

### 第一版范围

- 不做 compression。
- 只支持 pose retrieval。
- retrieved block 数固定。
- bank 存在 CPU，检索时搬到 GPU。

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

### 成功标准

- retrieval 可以通过配置启用/关闭。
- retrieved block 数为 0 时，行为应退化为原始 local window 或 sink-only。
- retrieved window token 数稳定。

## 里程碑 5：比较不同检索算法

### 目标

加入 retrieval metric 选择：

```text
recent_only
pose
fov
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

参考 HY-WorldPlay：

```text
distance = 1 - FOV_overlap(current block, historical block)
```

注意：使用 minWM 原生 pose convention，不要直接复制其他仓库里的 W2C/C2W 转换。

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

### 成功标准

- 不同 metric 能够选择不同的历史 block。
- 同一 seed 下检索结果可复现。
- 日志足够解释视频中的回环、一致性或漂移现象。

## 里程碑 6：加入 KV compression

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

两条分支最好共享同一组 token indices，确保 ordinary attention 和 PRoPE attention 看到的是对齐后的 token 子集。

### 成功标准

- compression 改变 token 数，但不改变 batch/head/dim 维度。
- compression 可以关闭。
- 分别测试 store-time compression 和 retrieval-time compression。

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

- dynamic pruning 降低平均 retrieved token 数。
- 相比固定 keep ratio，不明显损害 camera-following 稳定性。
- 在相近视觉质量下提升速度或降低显存。

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

早期 smoke test 可以先不做 RoPE correction，但正式对比必须补上。

### Token 数假设

compression 会改变 token 数。所有默认假设 `chunk_size * frame_seq_length` 的代码都需要检查。

### 显存与内存

初期优先使用 CPU KV bank。GPU bank 应该作为可选加速项，而不是默认路径。

### 实验归因

不要一开始同时加入 retrieval、compression、dynamic pruning。每次只加一个机制，避免结果无法解释。

## 近期建议步骤

1. 跑通 minWM Wan21 Action2V quickstart。
2. 加一个最小日志开关，记录 cache index 和显存。
3. 在 camera DMD config 中尝试 `sink_size: 4`。
4. fixed sink 确认后，再实现 periodic sink update。
5. 在写 retrieval 前，单独设计 KV bank API。

## 维护日志

- 2026-07-14
  - 创建路线规划。
  - 确认 minWM Wan21 是更适合单卡 1.3B 实验的平台。
  - 确认 minWM causal attention 具备 KV cache、PRoPE KV cache、`local_attn_size` 和潜在的 `sink_size` 机制。
  - 将文档改写为中文版本，方便后续持续维护。
  - 完成里程碑 1：加入可开关的 cache-state 日志，覆盖 causal DMD 与 causal diffusion 推理路径。
