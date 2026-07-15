# TartanAir Long5 小规模长视频测试集

## 用途

这个目录提供 5 条适合 minWM 长视频推理实验的 prompt 和 camera trajectory，用于研究 sink frame、KV cache 和后续 FOV/WorldKV retrieval 对长视频生成的影响。

## 当前版本

这是 starter benchmark：使用 TartanAir 风格的场景类型和 minWM 原生 trajectory 字符串，不下载原始 TartanAir RGB 或 ground-truth pose。这样可以先在当前 minWM 推理接口上快速跑通实验。

## 文件

- `prompts.txt`：5 条长视频 prompt，每行一个样本。
- `trajectories.txt`：5 条 camera trajectory，每行对应 `prompts.txt` 的同一行。
- `manifest.json`：样本元信息、轨迹长度和来源说明。

默认每条 trajectory 对应 `NUM_OUTPUT_FRAMES=120`。minWM 的 trajectory 规则是首帧为 identity，因此每条 trajectory 的操作步数总和为 `120 - 1 = 119`。

## 推理示例

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

DATA_PATH=Wan21/prompts/tartanair_long5_120/prompts.txt \
TRAJECTORY_PATH=Wan21/prompts/tartanair_long5_120/trajectories.txt \
NUM_OUTPUT_FRAMES=120 \
MAX_PROMPTS=5 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```
