# minWM × MBench-A 适配与测评指南

本文说明如何用 MBench-A 对 minWM 的 sink、retrieval 和 KV compression
策略进行统一评测。适配目标是保持所有 case 的样本、文本、相机轨迹、
输出长度、checkpoint 和 seed 完全相同，使最终差异只来自 memory 策略。

## 1. 为什么选择 MBench-A

minWM Wan Action2V 的输入是单条文本 prompt 加相机动作；MBench-A 也是
action-conditioned setting。MBench-T 需要五段随时间变化的文本条件，和当前
推理接口不一致，因此本适配只面向 `dataset_id: mbencha`。

MBench 评测的是生成视频，而不是 KV bank 本身。因而每个 minWM case 在
MBench 中注册成一个独立 `model_id`：

```text
minwm_baseline_seed0
minwm_pose_seed0
minwm_worldkv_fov_seed0
minwm_pose_compress_store_seed0
...
```

这样 MBench 的聚合结果可以直接比较不同 memory 策略。显存、耗时、retrieved
token 数和实际 keep ratio 仍应使用 minWM 的 profiling 文件报告，因为这些不在
MBench 指标范围内。

## 2. 适配做了什么

### 2.1 数据转换

`mbencha_adapter.py prepare` 遍历：

```text
MBench-A-Setup/samples/{subset}/{sample_id}/sample.json
```

读取 `caption`，并与指定 MBench-A condition 做笛卡尔组合，生成：

```text
adapter_root/
├── prompts.txt
├── trajectories.txt
└── manifest.jsonl
```

三者按 `prompt_index` 严格对齐。`manifest.jsonl` 保存
`subset/sample_id/condition_id`，供推理后反向包装结果。

### 2.2 动作语义转换

MBench-A action 到 minWM action 的映射为：

| MBench-A action | minWM trajectory |
|---|---|
| `left_then_right` | `j` 后接 `l` |
| `right_then_left` | `l` 后接 `j` |
| `forward_then_backward` | `w` 后接 `s` |
| `left_360/720/1080` | `j@倍率*N` |
| `right_360/720/1080` | `l@倍率*N` |
| `static` | `n*N` |

MBench 的 left/right 在 camera-interaction metric 中表示 yaw，而不是横向
平移，所以这里使用 `j/l`，没有使用 `a/d`。

minWM 原动作每步旋转 3°。为了在任意 rollout 长度下精确完成
360°/720°/1080°，轨迹解析器新增兼容语法：

```text
j@2.5*40
```

表示连续 40 步左转，每步使用基础转角的 2.5 倍。旧的 `w*19`、`j*40`
等语法保持不变。新增 `n*N` 用于静止或奇数长度往返轨迹的尾部补齐。

### 2.3 结果包装

推理结束后，adapter 读取每个 case 的 `inference_times.csv`，使用其中的
`prompt_index` 和实际 `output_path` 建立：

```text
MBench-A-Setup/models/{model_id}/
├── samples.jsonl
└── outputs/{subset}/{sample_id}/{condition_id}/video.mp4
```

默认使用相对软链接，不重复占用视频空间。需要将数据目录复制到其他机器时，
设置 `LINK_MODE=hardlink` 或 `LINK_MODE=copy`。

## 3. 环境准备

假设三个仓库/目录位置如下：

```text
research/
├── minWM/
├── MBench/
└── data/MBench-A-Setup/
```

先安装 minWM 和 MBench 环境。MBench-A 的完整 setup 必须包含
`dataset.yaml` 和 `samples/`；仓库中的 `MBench/demo/dataset/a` 仅适合验证
适配流程，不能代替正式数据集。

在 minWM 根目录设置：

```bash
export MBENCHA_ROOT=/absolute/path/to/MBench-A-Setup
export PYTHONPATH="$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"
```

## 4. 先做 dry-run

下面只准备两个 generation item 并打印各 case 的推理命令，不加载模型：

```bash
MBENCHA_ROOT=/absolute/path/to/MBench-A-Setup \
LENGTH=25s \
NUM_OUTPUT_FRAMES=400 \
CONDITIONS=left_then_right \
SUBSETS=environment \
LIMIT=2 \
CASES="baseline pose worldkv_fov pose_compress_store" \
SEEDS=0 \
DRY_RUN=1 \
bash Wan21/scripts/inference/run_mbencha_experiments.sh
```

确认以下文件正确：

```text
outputs/mbencha_adapter/25s/prompts.txt
outputs/mbencha_adapter/25s/trajectories.txt
outputs/mbencha_adapter/25s/manifest.jsonl
```

`NUM_OUTPUT_FRAMES` 必须满足当前 causal checkpoint 的 block size，默认要求
是 4 的倍数。它是 minWM 推理使用的帧/pose 数；应在所有 case 间固定。

## 5. 正式生成全部 case

推荐先分别运行 10s 和 25s，不要在同一 invocation 中混合长度：

```bash
MBENCHA_ROOT=/absolute/path/to/MBench-A-Setup \
LENGTH=25s \
NUM_OUTPUT_FRAMES=400 \
CONDITIONS=left_then_right,right_then_left,forward_then_backward,left_360,right_360 \
SUBSETS=environment,human,object,causal \
SEEDS="0 1 2" \
CASES="baseline fixed_sink pose pose_latent_frame worldkv_fov worldkv_fov_latent_frame hy_fov hybrid pose_compress_store worldkv_fov_compress_store worldkv_fov_dynamic_compress_store" \
RUN_ROOT=./outputs/mbencha_25s \
MODEL_PREFIX=minwm \
SKIP_COMPLETED=1 \
CONTINUE_ON_ERROR=1 \
bash Wan21/scripts/inference/run_mbencha_experiments.sh
```

注意：

1. `baseline` 和 `fixed_sink` 是必要控制组。
2. 首轮可只跑 `environment,object`，它们对空间回访与物体记忆最敏感。
3. 正式表格至少使用 3 个 seed，并分别报告均值和标准差。
4. 不要在 case 间修改 checkpoint、condition、帧数或采样参数。
5. `left_720/right_720/left_1080/right_1080` 也受支持，但只应在 MBench
   setup 确实定义/需要这些 condition 时加入。

如果推理已在其他方式下完成，可以单独包装：

```bash
python Wan21/scripts/evaluation/mbencha_adapter.py package \
  --dataset-root "$MBENCHA_ROOT" \
  --manifest outputs/mbencha_adapter/25s/manifest.jsonl \
  --run-root outputs/mbencha_25s \
  --model-prefix minwm \
  --link-mode symlink
```

## 6. MBench-A 校验与测评

进入 MBench 环境：

```bash
cd /absolute/path/to/MBench
pip install -e .
mbench list-metrics
```

先对一个 model、一个 metric 做严格校验：

```bash
mbench validate "$MBENCHA_ROOT" \
  --models minwm_worldkv_fov_seed0 \
  --metrics mbencha.entity.human_identity_consistency \
  --limit 2
```

无需 DA3 的 human/object-texture、lighting 和 prompt-interaction 指标可以
直接评测。示例：

```bash
mbench eval "$MBENCHA_ROOT" \
  --models minwm_baseline_seed0,minwm_pose_seed0,minwm_worldkv_fov_seed0,minwm_pose_compress_store_seed0,minwm_worldkv_fov_compress_store_seed0,minwm_worldkv_fov_dynamic_compress_store_seed0 \
  --metrics mbencha.entity.human_identity_consistency,mbencha.entity.human_appearance_consistency,mbencha.entity.object_texture_consistency,mbencha.environment.rendering_lighting,mbencha.causal.prompt_interaction \
  --output outputs/minwm_mbencha/no_da3 \
  --workers 1
```

以下指标依赖 DA3 `results.npz`：

```text
mbencha.entity.object_geometry_consistency
mbencha.environment.spatial_epipolar
mbencha.environment.spatial_reprojection
mbencha.environment.rendering_style
mbencha.causal.camera_interaction
```

按照 MBench README 使用外部 DA3 为每个生成视频制备：

```text
models/{model_id}/artifacts/{subset}/{sample_id}/{condition_id}/da3/results.npz
```

生成 artifact 后先执行 `mbench validate`，再运行：

```bash
mbench eval "$MBENCHA_ROOT" \
  --models minwm_baseline_seed0,minwm_pose_seed0,minwm_worldkv_fov_seed0,minwm_pose_compress_store_seed0,minwm_worldkv_fov_compress_store_seed0,minwm_worldkv_fov_dynamic_compress_store_seed0 \
  --metrics mbencha.entity.object_geometry_consistency,mbencha.environment.spatial_epipolar,mbencha.environment.spatial_reprojection,mbencha.environment.rendering_style,mbencha.causal.camera_interaction \
  --output outputs/minwm_mbencha/with_da3 \
  --workers 1
```

`state_progress` 和 `progress_correctness` 还需要按 MBench CLI 文档配置 VLM
judge；可在 VLM 服务就绪后加入：

```text
mbencha.causal.state_progress
mbencha.causal.progress_correctness
```

## 7. 最终结果如何报告

建议主表按 case 报告：

- MBench-A Entity、Environment、Causal 三组分数；
- 各组关键子指标；
- 3 个 seed 的 mean ± std。

效率表从 minWM 每个 case 输出目录收集：

- `inference_times.csv`：端到端耗时；
- `retrieval_events.csv/jsonl`：选择的历史 block、距离、retrieved token；
- profiling 日志：峰值显存；
- compression 记录：实际 keep ratio。

质量指标和效率指标需要联合解读。例如 compression 分数小幅下降但显存和
retrieved token 显著减少，属于有意义的 Pareto trade-off；只比较 MBench
总分无法回答 compression 是否有效。

## 8. 当前边界

- 本适配不会把 MBench 的 `first_frame.png` 输入 minWM，因为当前 Wan
  Action2V 实验接口是 text + camera action，而不是 first-frame-conditioned
  I2V。
- MBench-A 本身允许 action-conditioned 模型，因此这不妨碍使用其 A setting；
  但不要把该结果描述成 first-frame-conditioned 比较。
- MBench 只消费生成结果；它不会验证 KV retrieval 是否按预期发生。正式实验
  应同时检查 `retrieval_events`，避免把未触发 retrieval 的异常运行纳入结果。
