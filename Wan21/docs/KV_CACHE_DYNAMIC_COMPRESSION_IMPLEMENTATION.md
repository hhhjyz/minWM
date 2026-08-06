# minWM KV Cache 动态压缩与历史记忆实现说明

> 本文依据 2026-08-06 当前工作区代码整理，描述的是 **Wan21 causal camera inference 的实际实现**，包括未提交的工作区修改。核心代码位于 `pipeline/kv_bank.py`、`pipeline/retrieval_utils.py`、`pipeline/sink_utils.py`、`pipeline/causal_inference.py`、`pipeline/causal_diffusion_inference.py` 和 `wan/modules/causal_model.py`。

## 1. 结论先行

当前系统把长时历史拆成三类记忆：

1. **sink**：活动 KV cache 最前面的常驻帧，cache 滚动时不会被淘汰；
2. **retrieved memory**：从 sidecar KV bank 按相机位姿或 FOV 相似度选出的历史块/帧；
3. **recent/local memory**：活动 cache 中 sink 之后的最近历史。

启用 retrieval 后，普通 attention window 由纯 local window 改为：

```text
[ sink K/V | retrieved K/V | recent K/V ]  <- current Q
```

总 attention token budget 不变；retrieved token 越多，recent token 就越少。KV bank 保存每个生成 block 在 clean-context rerun 后得到的 normal RoPE KV，以及存在相机条件时的 PRoPE KV。压缩可以在写入 bank 时执行一次，也可以在每次取出 payload 时执行。

所谓“动态调整压缩率”目前仅影响 **store-time compression**。其控制信号是 block 首尾相机位姿的平移与旋转幅度，实际公式是：

```text
motion = translation_delta / translation_scale
       + rotation_delta_rad / rotation_scale

keep_ratio = clamp(base_keep_ratio - motion_weight * motion,
                   min_keep, max_keep)
```

因此当前实现的语义是：**相机运动越大，keep ratio 越低，压缩越强**。这一方向与“运动越大、场景变化越多、应保留更多 token”的常见直觉相反，后续优化前应先确认这是不是设计意图。

## 2. 实现范围与入口

这套机制目前接入两条 causal pipeline：

- `CausalInferencePipeline`：少步 ODE/CD/DMD 推理；
- `CausalDiffusionInferencePipeline`：causal diffusion 推理。

两者的 KV bank、retrieval、compression 和 sink 调用逻辑基本平行。bidirectional pipeline 没有接入 `KVBank`，HY15 目录也没有这套动态压缩实现。

配置链为：

```text
scripts/inference/run_infer_causal_camera.sh 环境变量
    -> wan_inference.py CLI 参数
    -> OmegaConf config / model_kwargs
    -> causal pipeline
    -> CausalWanModel / CausalWanSelfAttention
```

模型中固定 `frame_seq_length = 1560`，即一个 latent frame 对应 1560 个 spatial tokens。文中“frame”均指模型 latent frame，不是最终视频的原始像素帧。

## 3. 一次 block 的完整生命周期

对当前 block，实际顺序是：

1. 根据当前相机 `viewmats` 从已有 KV bank 选择历史 block 或历史 latent frame；
2. 为每个 transformer layer 构造 retrieval payload，必要时做 runtime compression、normal RoPE correction 或 PRoPE re-encoding；
3. 在各 denoising step 中，以同一份 retrieval payload 参与 attention；
4. 得到当前 block 的 `denoised_pred`；
5. 用 clean/低噪声 context 再 forward 一次，更新活动 KV cache；
6. 从活动 cache 尾部抽取刚写入的 clean KV，追加到 sidecar KV bank；
7. 若达到更新间隔，再执行 periodic/bank sink 替换。

这一顺序很重要：当前 block 不能检索自身；bank 中存的是 clean-context rerun 的 KV；sink 更新发生在 bank append 之后。

## 4. 活动 KV cache 与 sink frame

### 4.1 活动 cache 的布局与滚动

每个 attention layer 的 normal/PRoPE cache 至少包含：

- `k`, `v`：形状近似 `[B, cache_tokens, heads, head_dim]`；
- `global_end_index`：真实时间轴已经处理到的位置；
- `local_end_index`：当前物理 cache 有效区末端。

启用有限 local attention 且空间不足时，cache 会左移淘汰旧 token，但移动范围从 `sink_tokens` 开始：

```text
[0 : sink_tokens]               永不参与滚动淘汰
[sink_tokens : local_end_index] 滚动 local 区
```

`sink_tokens = sink_size_frames * frame_seq_length`。CLI 会检查 `sink_size < local_attn_size`。

### 4.2 sink 策略

| 策略 | 当前行为 |
|---|---|
| `none` | sink 大小强制设为 0 |
| `fixed` | 首批写入 cache 的前 `sink_size` 帧永久保留 |
| `periodic` | 每 N 个生成 block，把活动 cache 中最新 token 原样复制到 sink 槽 |
| `bank_random` | 从合法 bank 候选中确定性伪随机选一个 block 覆盖 sink |
| `bank_uniform` | 按更新次数对候选列表循环取样 |
| `bank_pose` | 用 pose retrieval 选排名第一的 block 覆盖 sink |
| `bank_worldkv_fov` | 用 WorldKV FOV retrieval 选排名第一的 block覆盖 sink |

`bank_fov` 只是 `bank_worldkv_fov` 的别名。

除 `fixed` 外的更新策略都要求 `sink_update_interval > 0`，并在 `(block_index + 1) % interval == 0` 时触发。

### 4.3 sink 候选过滤

bank sink 候选必须：

- 包含所需 branch；
- `block.frame_start >= sink_size_frames`，避免再次选择初始固定 sink 区；
- `block.frame_end <= current_frame_start`，不能来自未来或当前未完成区域；
- 若设置 recent exclusion，还需 `block.frame_end <= current_frame_start - recent_frames`。

### 4.4 sink 更新的几何语义

`periodic` 直接复制已经编码过 RoPE/PRoPE 的 K/V，不重新编码位置或相机几何。这是代码注释明确标注的 v0 近似。

bank sink 同样默认直接复制 bank 中的编码后 K/V。若 bank block 因压缩而少于 sink budget，剩余 sink 槽会清零，防止残留上一次 sink 内容。注意 attention 仍按完整 `sink_tokens` 切片，因此这些清零槽仍占物理 attention budget。

`prope_reencode_mode=current` 可尝试把未压缩 bank block 的 PRoPE KV 从源相机重新编码到当前相机；store-time compressed block 明确跳过此操作，因为压缩后 token 已不再保持规则的逐帧空间布局。

## 5. KV bank

### 5.1 bank block 数据结构

每个 `KVBankBlock` 保存：

- `block_id`, `[frame_start, frame_end)`；
- 原始 `token_count = frame_count * frame_seq_length`；
- 每个 branch、每个 layer 的 normal `k/v` 和可选 PRoPE `k/v`；
- CPU float32 的 `viewmats`、`Ks`；
- block 首尾相机运动摘要；
- 实际 storage bytes；
- 是否 store-time compressed、实际 keep ratio、motion score。

bank 默认存 CPU，也可存 CUDA。所有张量均 `detach -> to(device) -> clone`。`max_blocks=0` 表示无限；正数表示追加前 FIFO 淘汰最老 block。

bank 会输出首 block 大小、预计总占用和超阈值警告；`summary()` 暴露 block 数、淘汰数、总字节数和 compression 配置。

### 5.2 保存时从哪里取 KV

完成 clean-context rerun 后，`append_block()` 根据当前 block 的原始 token 数，从每层活动 cache 的尾部抽取：

```text
local_end = cache.local_end_index
slice = cache[:, local_end - token_count : local_end]
```

normal KV 与 PRoPE KV 分别抽取和保存。相机输入存在时保存 PRoPE branch；当前 pipeline 的 branch 名是 `main`。

## 6. Retrieval 选择

### 6.1 粒度

支持两种粒度：

- `chunk`：选择完整 bank block；
- `latent_frame`：在 block 内逐帧建立候选并切取完整的 1560-token frame。

`latent_frame + store-time compression` 被显式禁止，因为压缩后逐帧 token 不再是固定的 1560-token 规则切片。latent-frame retrieval 当前也没有 runtime compression 分支。

### 6.2 候选过滤

chunk retrieval 排除：

- `block.frame_start < sink_size_frames` 的初始 sink 区；
- 与 recent exclusion 重叠的 block；
- 尚未完整位于当前时间之前的 block；
- 没有相机 metadata 的 block。

latent-frame retrieval 对每一帧做等价过滤。

### 6.3 `retrieval_frames` 的真实语义

latent-frame 模式会精确 top-k 到最多 `retrieval_frames` 帧。

chunk 模式先计算：

```text
num_blocks = ceil(retrieval_frames / current_frame_count)
```

再返回这么多个完整 block。因此 `retrieval_frames` 在 chunk 模式是“换算 block 数的目标帧预算”，不是严格的返回帧数；历史 block 大小若和当前 block 不同，实际返回帧数可能偏离目标。

### 6.4 距离指标

#### Pose

每个 block 先把 W2C 取逆得到 C2W，再对 block 内相机中心和平移/旋转矩阵取均值：

```text
translation_distance = squared L2(mean_t_hist - mean_t_current)
rotation_distance    = acos((trace(R_hist^T R_current) - 1) / 2)
distance = 0.5 * normalize(translation_distance)
         + 0.5 * normalize(rotation_distance)
```

归一化是候选集合内除以最大值，因此分数依赖当次候选集合。

#### WorldKV FOV

确定性地用 golden-angle 方式在指定半径球内生成约均匀 probe points。chunk 只比较当前与历史 block 的中间帧：先把当前相机坐标点变到世界坐标，再测试其是否也落在历史相机 frustum 与半径内：

```text
similarity = intersection_visible_points / current_visible_points
distance   = 1 - similarity
```

默认 8192 点、半径 8、水平 FOV 60°、垂直 FOV 35°。

#### HY FOV

同样比较 block 中间帧，但使用相对 W2C、pitch/yaw 和方位角/仰角范围测试，最后仍取 `1 - overlap`。

#### Hybrid

```text
distance = (1-w) * normalize(pose_distance) + w * worldkv_fov_distance
```

`w` 会 clamp 到 `[0, 1]`。

#### recent_only

当前实现中 `recent_only` 直接返回空 selection，等价于“不注入 retrieved KV”，并不是从 bank 中选择最近 block。recent memory 仍由活动 local cache 自然提供。

### 6.5 排序与时序

候选按 `(distance, bank_index)` 升序排序，距离相同时较老的 bank index 优先。被选 payload 的拼接顺序是“相似度排名顺序”，不是源时间顺序。

## 7. KV compression

### 7.1 压缩发生位置

有两种模式：

1. **store-time**：`compression_enable && compression_at_store`，每个 block 写 bank 时每层压缩一次；节省 bank 内存和后续 CPU/GPU 传输；
2. **runtime/retrieval-time**：启用 compression 但关闭 `at_store`，bank 保存完整 KV，每次构建 chunk retrieval payload 时重新压缩。

动态 keep ratio 只在 `append_block()` 中计算并传给 store-time compression。若打开 dynamic 但未打开 `compression_at_store`，motion score 虽会计算，但 runtime compression 仍固定使用全局 `kv_compression_keep_ratio`，动态配置不会改变实际 token 数。

### 7.2 anchor + novelty 算法

输入必须满足：

```text
total_tokens % (chunk_size * frame_seq_length) == 0
```

否则函数静默返回未压缩 KV。每个 chunk reshape 为：

```text
[B, frames_in_chunk, tokens_per_frame, heads, dim]
```

anchor frame 完整保留。默认 anchor 是 chunk 的第 0 帧；runtime 模式可通过 `anchor_rotate` 令第 `chunk_index % chunk_size` 帧成为 anchor。store-time 调用硬编码 `anchor_rotate=False`，所以该参数只影响 runtime compression。

anchor 的所有 spatial token 在 token 维取均值，得到 `[B, heads, dim]` centroid。对其他每个 token，跨全部 heads 和 head_dim 计算其与 centroid 的 cosine similarity，选择 similarity **最小** 的 token，即相对 anchor 最“新颖”的 token。

每个非 anchor frame 的预算是：

```text
keep_per_frame = max(1, ceil(keep_ratio * frame_seq_length))
```

若 `keep_per_frame >= frame_seq_length`，整个输入原样返回。选择后会按原 token index 排序，以保留空间顺序。

### 7.3 pooled 与 per-frame

- `pooled=False`：每个非 anchor frame 独立保留 `keep_per_frame` 个 token；
- `pooled=True`：把所有非 anchor token 合并，统一选 `(F-1) * keep_per_frame` 个 novelty token，允许不同帧得到不同预算。

两者总 token 数相同：

```text
compressed_tokens_per_chunk = L + (F - 1) * ceil(r * L)
effective_total_keep = compressed_tokens_per_chunk / (F * L)
```

其中 `F=chunk frames`、`L=1560`、`r=keep_ratio`。所以 CLI 的 keep ratio 只针对非 anchor frame；总保留率总是更高。例如 `F=4, r=0.5` 时保留 `1560 + 3*780 = 3900`，总保留率为 62.5%。

K 用来计算 novelty index，同一 index 同步 gather V。normal KV 与 PRoPE KV 分别独立运行压缩，因此二者可能选择不同的 token index；它们是两条独立 attention 路径，不要求逐 token 对齐。

### 7.4 动态 keep ratio

相机摘要来自 block 的首尾 C2W：

```text
translation_delta = mean_batch(||t_end - t_start||_2)
rotation_delta_rad = mean_batch(acos((trace(R_start^T R_end)-1)/2))
```

随后：

```text
motion_score = translation_delta / max(translation_scale, 1e-6)
             + rotation_delta_rad / max(rotation_scale, 1e-6)

effective_keep_ratio = clamp(
    base_keep_ratio - motion_weight * motion_score,
    dynamic_min_keep,
    dynamic_max_keep,
)
```

默认值：base=0.5、min=0.25、max=0.75、translation scale=1.0、rotation scale=0.35 rad、weight=0.25。

几个容易忽略的边界行为：

- 无 `viewmats` 时动态逻辑退化为 base keep ratio，`motion_score=None`；
- 即使静止，base 也会被 clamp 到 `[min,max]`；
- motion score 没有上限，keep ratio 最终由 min 截断；
- 参数没有检查 `min_keep <= max_keep`，反向配置会得到不直观结果；
- keep ratio 没有统一 clamp 到 `[0,1]`，但负值最终仍因 `max(1, ceil(...))` 每帧至少保留 1 token，大于等于 1 则完全不压缩；
- `pooled=True` 且 block 只有一个 frame、同时确实需要压缩时，会尝试拼接空的 non-anchor 列表，存在运行时错误风险。

## 8. Attention window 与位置修正

### 8.1 默认 retrieval window

对每层，模型先更新活动 cache，再组装：

```text
sink_end     = min(sink_tokens, local_end)
retr_tokens  = retrieved_k.shape[1]
recent_budget = max_attention_tokens - sink_end - retr_tokens
recent       = cache tail within recent_budget, excluding sink
window       = [sink, retrieved, recent]
```

若 retrieved token 已超过剩余 attention budget，`recent_budget` 变为 0，但代码不会裁剪 retrieved KV，因此最终 window 可能超过 `max_attention_size`。

没有 retrieval 时，普通路径直接取活动 cache 最后 `max_attention_size` 个 token；这一路不会显式拼回 sink。因此仅设置 sink 但 local window 已滚动很长时，是否实际 attend 到 sink 取决于其他 rebase/window 路径；当前普通 no-retrieval 分支不保证 `[sink|recent]` 组合。

### 8.2 normal RoPE correction

`retrieval_rope_correction` 只修正 retrieved normal K，不修正 V。它根据每段 payload 的源 `src_frame_id` 与其在 recent 前方的虚拟时间位置，给 K 乘时间轴 RoPE delta。`chunk_token_lengths` 用于支持压缩后的可变 token 段。

### 8.3 fixed-sink-only rebase

该模式要求 `sink_strategy=fixed` 且 retrieval 关闭。local K 和 current Q 保持真实时间位置，只把 sink K 映射到 local window 前方连续位置：

```text
[rebased sink | real-time local/current]
```

它在 `current_start_frame >= sink_size` 后启用。

### 8.4 tri-region bounded RoPE

该模式要求 fixed sink。warm-up 阶段保持普通单调 RoPE，并在 `current_start_frame < rope_train_length` 时由 pipeline 同步禁用 retrieval。达到边界后：

- current Q 的最后一帧映射到 `rope_train_length`；
- local K（最近 `rope_local_window` 帧加 current chunk）映射到 Q 前方/相同的连续区；
- retrieved K 从 sink 之后开始映射；
- sink 保持起始区域；
- 代码检查 retrieval virtual span 不能与 local 区重叠。

默认 `sink=4, retrieval span=8 frames, local previous=4, current chunk=4, T=19` 时意图布局是：

```text
sink: 0..3 | retrieval: 4..11 | recent+current: 12..19
```

这里的 retrieval span 由 `len(src_frame_ids) * compress_chunk_size` 估算，而不是由压缩后实际 token 数推断。因此压缩减少 token 后仍保留原 chunk 的虚拟时间跨度。

PRoPE retrieval 路径不会进入 normal K 的 tri-region/time-RoPE rebase；它使用 payload 中的 `prope_k/prope_v`，可选通过相机投影矩阵做 `current` re-encoding。

## 9. PRoPE re-encoding

bank 中存储的 PRoPE KV被视为近似 `P_src_inv @ raw_kv`。目标相机重编码使用：

```text
transform = P_dst_inv @ P_src
reencoded = transform @ stored_feature
```

仅在以下条件满足时执行：

- source/destination 都有 viewmats；
- 相机帧数一致；
- token 数能整除相机数，且 feature dim 能被 4 整除；
- source/destination 同时有 K 或同时没有 K；
- bank block 未做 store-time compression。

chunk retrieval 需要源 block 与当前 block 的 camera count 一致；latent-frame retrieval 使用源单帧和当前 block 中间帧。失败时静默保留原 PRoPE KV，并通过 `prope_reencoded=false` 记录。

## 10. 当前实验预设的实际行为

`run_string_camera_experiments.sh` 定义了主要 case：

- `fixed_sink`、`periodic_sink` 和四种 bank sink；
- `pose/worldkv_fov/hy_fov/hybrid` retrieval；
- `pose_latent_frame/worldkv_fov_latent_frame`；
- `pose_compress_store/worldkv_fov_compress_store`：store-time、pooled compression；
- `worldkv_fov_dynamic_compress_store`：在上一项基础上启用 motion-driven ratio。

当前脚本还会对多数 retrieval case 自动启用 fixed sink + tri-region，并把 `retrieval_frames` 覆盖为 `TRI_REGION_RETRIEVAL_FRAMES`。因此 case 分支里设置的 `COMPRESSED_RETRIEVAL_FRAMES` 或 `DYNAMIC_COMPRESSED_RETRIEVAL_FRAMES` 可能随后被统一 tri-region 配置覆盖；做消融时必须查看 dry-run 输出确认最终参数，而不能只看 case 分支。

## 11. 日志与可观测性

### 11.1 bank 日志

`[kv-bank]` 每次记录：原始 frame/token 范围、bank block 数、总内存、淘汰数、store-time effective keep ratio 和 motion score。注意日志中的 `tokens=` 是压缩前 `token_count`，不是实际存储 token 数。

### 11.2 retrieval event

每个有资格执行 retrieval 的 block 都记录：

- current frame start/count；
- metric、granularity、目标 retrieval frames；
- candidate/selected block IDs、selected frame starts、全部候选 distances；
- 每层实际 retrieved token 数；
- selection 与 payload 构建耗时；
- RoPE correction、PRoPE re-encoding 状态。

事件写到输出目录的 `retrieval_events.jsonl` 和 `retrieval_events.csv`。tri-region warm-up 被提前 return，不会记录“未检索”事件。

### 11.3 当前缺失的关键统计

现有 event 没记录 selected distance、每个 selected block 的实际 keep ratio/motion score、实际 retrieved latent-frame 等价预算、recent token 数和最终 attention window 长度。这些是分析动态压缩收益时最需要补齐的指标。

## 12. 已确认的限制与潜在优化点

建议按优先级考虑：

1. **确认动态方向**：当前 `base - weight*motion` 让大运动更强压缩；若目标是保留新信息，可能应改为加号、反向 motion，或直接用内容 novelty/重访概率驱动。
2. **让动态策略覆盖 runtime compression**：目前 dynamic 开关在非 store-time 模式下实际不控制压缩比例。
3. **按检索价值分配预算**：当前 ratio 在存储时决定，尚不知道该 block 将来是否重要；可保存完整/轻量摘要，在 retrieval 时结合 distance、运动、时间和总 attention budget 二次分配。
4. **动态 `retrieval_frames` 与 keep ratio 联合优化**：固定块数 × 固定比例不能保证固定 token budget；应直接以 token budget 为约束。
5. **明确 motion 的物理尺度**：平移和旋转简单相加，trajectory 尺度变化会直接改变策略；可使用场景尺度、速度、FOV 位移或光流归一化。
6. **使用 block 内完整轨迹**：当前动态压缩只看首尾位姿，往返运动可能首尾接近而被误判为静止；可累计逐帧 path length 与 rotation arc。
7. **内容感知压缩**：当前 novelty 只相对 anchor K centroid，未利用 V、query、attention score、跨层一致性、空间 coverage 或语义区域。
8. **保护 sink/位置一致性**：periodic/bank sink 的编码位置近似、compressed bank sink 的零填充、normal 与 PRoPE 独立选 token，都值得专项消融。
9. **严格 attention budget**：对 retrieved token 超预算进行裁剪或预算分配，避免 window 超出预期；同时修正普通 no-retrieval sink 不保证进入 window 的问题。
10. **统一时序排序策略**：当前按相似度顺序拼接，RoPE virtual slots 也按排名分配；应对比“相似度选择后按源时间排序”。
11. **缓存 FOV probe 与 block descriptor**：当前每次 selection 都在 CPU 重新生成 probes、重新计算所有候选几何距离，长视频下复杂度随 bank 线性增长。
12. **修正边界与校验**：增加 keep ratio 范围、min/max 顺序、单帧 pooled、不同 block frame count、不同压缩长度、retrieved token 超预算等检查。
13. **改善日志**：记录每块压缩前后实际 token/bytes、selected block 的 ratio/motion、最终 `[sink,retrieved,recent]` 长度，并形成质量-显存-延迟 Pareto 曲线。

## 13. 推荐的优化实验矩阵

为了判断收益来自“检索”“压缩”还是“位置处理”，建议固定 prompt、trajectory、seed、checkpoint 与总 attention token budget，至少比较：

| 组别 | Retrieval | Compression | RoPE/PRoPE |
|---|---|---|---|
| A | 无 | 无 | ordinary local |
| B | fixed sink，无 retrieval | 无 | ordinary / sink-only rebase |
| C | FOV/pose | 无 | correction off/on，tri-region |
| D | 同 C | 固定 store-time ratio | pooled off/on |
| E | 同 C | 当前动态减法公式 | 记录 motion-ratio 曲线 |
| F | 同 C | 动态加法或可学习映射 | 固定总 retrieved token budget |
| G | 同 C | retrieval-time 联合预算 | distance + motion + novelty |

核心横轴应使用 **实际 retrieved tokens / 最终 attention tokens / bank bytes**，而不是只使用名义 `retrieval_frames` 和非 anchor `keep_ratio`。

## 14. 核心代码索引

- `Wan21/wan_inference.py`：CLI、参数约束、配置注入；
- `Wan21/pipeline/causal_inference.py`：少步 causal 推理的 retrieval/build/store/update 调度；
- `Wan21/pipeline/causal_diffusion_inference.py`：diffusion 对应调度；
- `Wan21/pipeline/kv_bank.py`：bank 数据结构、动态比例、compression、block/frame retrieval payload；
- `Wan21/pipeline/retrieval_utils.py`：pose、WorldKV FOV、HY FOV、hybrid 距离；
- `Wan21/pipeline/sink_utils.py`：periodic/bank sink 更新；
- `Wan21/wan/modules/causal_model.py`：cache 滚动、attention window 组装、RoPE correction 与 tri-region；
- `Wan21/scripts/inference/run_infer_causal_camera.sh`：环境变量到 CLI；
- `Wan21/scripts/inference/run_string_camera_experiments.sh`：实验 case 的最终组合逻辑。
