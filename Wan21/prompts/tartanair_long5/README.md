# TartanAir Long5 小规模长视频测试集

## 用途

这个目录提供 5 条适合 minWM 长视频推理实验的 prompt 和 camera trajectory，用于研究：

- sink frame 是否改善长时一致性；
- KV cache 随时间滚动时的稳定性；
- 后续 WorldKV/FOV retrieval 接入后的长视频对比。

## 数据集选择

选择 TartanAir 作为参考数据集，原因是它是合成环境，场景类型丰富，真实版本包含干净的 ground-truth camera pose，适合做几何检索和长时 memory 实验。

当前版本是 starter benchmark：使用 TartanAir 风格的场景类型和 minWM 原生 trajectory 字符串，不下载原始 TartanAir RGB 或 GT pose。这样可以先在当前 minWM 推理接口上快速跑通实验。后续可以扩展为真实 TartanAir pose 读取。

## 文件

- `prompts.txt`：5 条长视频 prompt，每行一个样本。
- `trajectories.txt`：5 条 camera trajectory，每行对应 `prompts.txt` 的同一行。
- `manifest.json`：样本元信息、轨迹长度和来源说明。

默认每条 trajectory 对应 `NUM_OUTPUT_FRAMES=80`。minWM 的 trajectory 规则是首帧为 identity，因此每条 trajectory 的操作步数总和为 `80 - 1 = 79`。

## 重新生成

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

conda run -n ling python Wan21/scripts/data_preprocessing/build_tartanair_long5.py \
  --num-frames 80 \
  --output-dir Wan21/prompts/tartanair_long5
```

如果要生成更长的视频，例如 120 帧：

```bash
conda run -n ling python Wan21/scripts/data_preprocessing/build_tartanair_long5.py \
  --num-frames 120 \
  --output-dir Wan21/prompts/tartanair_long5_120
```

推理时需要同步设置：

```bash
DATA_PATH=Wan21/prompts/tartanair_long5_120/prompts.txt \
TRAJECTORY_PATH=Wan21/prompts/tartanair_long5_120/trajectories.txt \
NUM_OUTPUT_FRAMES=120 \
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```
