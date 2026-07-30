# 10 秒闭环测试：World Retrieval × World Compression 预算消融实验

本实验在 10 秒 loop-closure 测试集上，对 retrieval 历史覆盖范围、固定
KV 压缩和相机运动自适应压缩进行六组对比。

## 统一实验设置

- Prompt：`Wan21/prompts/demos_loop_closure/prompts.txt`
- 相机轨迹：`trajectories_10s.txt`
- 输出长度：40 个 latent frames
- 每个生成 chunk：4 个 latent frames
- 每个 latent frame：1560 个视觉 token
- Local attention 上限：20 frame-equivalents，即 31,200 tokens
- Sink：固定保留最初 4 帧
- Retrieval 粒度：完整的 4-frame chunk
- Retrieval 指标：`worldkv_fov`
- Retrieval recent exclusion：8 帧
- 压缩时机：KV block 写入 KV bank 时压缩一次
- 非 anchor token：跨所有非 anchor 帧统一 pooled selection
- Tri-region RoPE：本组实验关闭

本实验关闭 tri-region RoPE，是因为当前实现会为每个原始 retrieval frame
分配一个虚拟 RoPE 位置。即使 compression 已经降低物理 token 数，12 帧和
16 帧 retrieval coverage 仍无法放入有限的虚拟 memory 区域。

关闭 tri-region 后，普通 retrieval attention 路径仍会把总 attention
限制在模型原生的 31,200-token 上限内，因此可以进行固定物理 attention
预算实验。

## Token 预算计算

minWM 的每个 chunk 包含 4 个 latent frames。Compression 完整保留第一帧
anchor，并从其余 3 帧中分别或统一保留比例为 `r` 的 token：

```text
单个压缩 chunk 的 frame-equivalent = 1 + 3r
```

| 组别 | 原始 retrieval coverage | Chunk 数 | 压缩设置 | 单 chunk 等价大小 | Retrieval 物理预算 |
|---|---:|---:|---:|---:|---:|
| A | 8 帧 | 2 | 不压缩 | 4.0 | 8.0 |
| B | 8 帧 | 2 | r=0.5 | 2.5 | 5.0 |
| C | 12 帧 | 3 | r=0.5 | 2.5 | 7.5 |
| D | 16 帧 | 4 | r=1/3 | 2.0 | 8.0 |
| E | 16 帧 | 4 | 动态，移动越快保留越多 | 1.6–2.5 | 6.4–10.0 |
| F | 16 帧 | 4 | 动态，移动越快压缩越强 | 1.6–2.5 | 6.4–10.0 |

所有实验的物理 attention 上限均为 20 frame-equivalents。当 retrieval
占用的 token 较少时，普通 attention 组窗逻辑会用更多 recent KV 填充剩余
容量。

## 各实验组设置与目标

### A — `A_retr8_no_compression`

检索 8 个原始历史帧，不进行压缩。该组是只启用 World Retrieval 的基准，
retrieval 占用 8 个 frame-equivalents。

### B — `B_retr8_compression_r050`

检索与 A 完全相同的 8 个原始历史帧，但将其压缩到 5 个
frame-equivalents。

A 与 B 的比较用于回答：

> 在历史覆盖范围相同时，compression 会造成多少信息损失，又能节省多少
> 存储和 attention 计算？

这不是固定 retrieval token budget 的比较，因为 B 的物理 retrieval token
明显少于 A。

### C — `C_retr12_compression_r050`

检索 12 个原始历史帧，并以 `r=0.5` 压缩到 7.5 个
frame-equivalents。

A 与 C 的比较用于测试：

> 在近似相同的 retrieval token 预算下，将历史覆盖扩大到 1.5 倍是否更有利？

由于 retrieval 必须以完整 4-frame chunk 为单位，`r=0.5` 时无法恰好得到
8 个 frame-equivalents。C 的 retrieval 预算比 A 低 6.25%。

### D — `D_retr16_compression_r0333`

检索 16 个原始历史帧，并以 `r=1/3` 精确压缩到 8 个
frame-equivalents。

A 与 D 是本实验最重要的 WorldKV 风格对比：

```text
A：8 个原始帧  → 8 个 frame-equivalents
D：16 个原始帧 → 8 个 frame-equivalents
```

两组具有相同的物理 retrieval token 预算，但 D 的历史覆盖范围是 A 的
两倍。对于 minWM 的 4-frame chunk，`r=1/3` 对应论文中的 2× 压缩设置。

### E — `E_retr16_dynamic_fast_keep_more`

检索 16 个原始历史帧，并使用以下动态规则：

```text
motion_score =
    translation_delta / 1.0
  + rotation_delta_rad / 0.35

keep_ratio = clamp(
    0.25 + 0.25 × motion_score,
    0.2,
    0.5
)
```

当前代码内部公式为：

```text
keep = base - motion_weight × motion_score
```

因此该组通过设置 `motion_weight=-0.25` 实现“移动越快，保留越多 token”。

该组检验的假设是：

> 快速相机运动会揭示更多新区域和非冗余内容，因此应该降低压缩强度、保留
> 更多 token。

### F — `F_retr16_dynamic_fast_compress_more`

同样检索 16 个原始历史帧，但采用相反的动态方向：

```text
motion_score =
    translation_delta / 1.0
  + rotation_delta_rad / 0.35

keep_ratio = clamp(
    0.4167 - 0.25 × motion_score,
    0.2,
    0.5
)
```

相机移动越快，compression 越激进。这是项目原始动态压缩公式采用的方向。

该组检验的假设是：

> 快速经过的中间视角不需要保存过多稠密信息，可以更激进地压缩；缓慢运动
> 或停留视角则应保存更多细节。

## 为什么 E 和 F 使用不同的 base keep ratio

此前 10 秒 loop-closure 实验中，观测到的平均 motion score 约为 `0.323`。
E 和 F 的 base 值经过设置，使两种动态方向的预期平均 keep ratio 都接近
`1/3`：

```text
E：0.25   + 0.25 × 0.323 ≈ 0.331
F：0.4167 - 0.25 × 0.323 ≈ 0.336
D：固定 r = 1/3              ≈ 0.333
```

这样可以尽量让 D、E、F 的平均物理 retrieval token 预算一致，主要比较
token 在不同相机速度区间中的分配方式。

E 和 F 都将 keep ratio 限制在 `[0.2, 0.5]`。对于 4 个 retrieved chunks：

```text
r=0.2 → retrieval 约占 6.4 frame-equivalents
r=0.5 → retrieval 约占 10.0 frame-equivalents
```

该范围可以避免 retrieval 占用过多 token，导致当前 4-frame query chunk
被排除在 20-frame attention budget 之外。

这里的预算匹配只是根据历史 motion score 分布得到的近似值。正式比较 D、
E、F 前，必须检查实际生成得到的平均 keep ratio 和 retrieval token 数。

## 主要比较关系

| 对比 | 回答的问题 |
|---|---|
| A vs B | 相同历史覆盖下，compression 本身造成多少信息损失？ |
| A vs C | 近似固定预算下，1.5× 历史覆盖是否更好？ |
| A vs D | 严格固定预算下，2× 历史覆盖是否更好？ |
| B vs C | 是否应该把 compression 节省的 token 用于扩大历史覆盖？ |
| D vs E | 快速运动时保留更多 token 的动态分配是否优于固定比例？ |
| D vs F | 项目原始的“快速运动时压缩更多”规则是否优于固定比例？ |
| E vs F | 相机速度自适应压缩应该采用哪个方向？ |

## 动态压缩结果分析要求

当前 motion score 测量固定 4-frame chunk 内的相机位移。由于每个 chunk 的
持续时间相同，它与相机移动速度成比例，但不是经过物理单位标定的真实速度。

平移和旋转分别使用以下归一化尺度：

```text
translation scale = 1.0
rotation scale    = 0.35 radians，约 20 度
```

分析动态压缩时，需要同时报告质量和实际预算：

1. Loop-closure LPIPS、PSNR 和 SSIM；
2. Effective keep ratio 的均值与分布；
3. 每层平均 `retrieved_tokens_per_layer`；
4. KV bank 最大存储量；
5. 视频生成耗时；
6. 如果可以获得样本级 motion metadata，分别统计慢速和快速运动样本。

如果 E 或 F 的平均 retrieval token 数与 D 存在明显差异，应先重新校准 base
keep ratio，再把质量差异解释为动态分配带来的收益。

## 输出目录结构

每个实验组包含生成视频、计时文件、retrieval event、profiling 结果和闭环
评测结果：

```text
A_retr8_no_compression/
B_retr8_compression_r050/
C_retr12_compression_r050/
D_retr16_compression_r0333/
E_retr16_dynamic_fast_keep_more/
F_retr16_dynamic_fast_compress_more/
```

每组的主要评测文件：

```text
eval/loop_closure_metrics.csv
eval/loop_closure_metrics.json
compression_budget_summary.json
```

实验根目录还会生成跨组预算汇总：

```text
compression_budget_summary.csv
```

在比较 D、E、F 的 loop-closure 指标之前，应先用该文件确认三组的实际平均
retrieval token 预算接近。

所有组使用相同的 prompt index、seed 和输出文件名，可以直接进行同名视频的
配对观察。

## 运行方法

在 minWM 项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
bash Wan21/scripts/inference/run_loop10s_worldkv_compression_budget_ablation.sh
```

单样本 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PROMPTS=1 \
RUN_ROOT=./outputs/loop10s_worldkv_compression_budget_ablation_smoke \
bash Wan21/scripts/inference/run_loop10s_worldkv_compression_budget_ablation.sh
```

脚本默认设置 `SKIP_COMPLETED=1`，中断后重新运行会跳过已经完整生成的实验组。

如需进行不计算 LPIPS 的轻量评测：

```bash
EVAL_LOOP_SKIP_LPIPS=1 \
bash Wan21/scripts/inference/run_loop10s_worldkv_compression_budget_ablation.sh
```
