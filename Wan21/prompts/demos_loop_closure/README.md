# Demo Loop-Closure Trajectories

这个目录为 `Wan21/prompts/demos.txt` 中的 30 条 prompt 提供四种长度的
minWM action-string 相机轨迹。

| 文件 | Latent poses | 解码帧数 | 16 FPS 时长 | Loop closure pair |
|---|---:|---:|---:|---:|
| `trajectories_10s.txt` | 40 | 157 | 9.8125 s | `(0, 38)` |
| `trajectories_15s.txt` | 60 | 237 | 14.8125 s | `(0, 58)` |
| `trajectories_20s.txt` | 80 | 317 | 19.8125 s | `(0, 78)` |
| `trajectories_30s.txt` | 120 | 477 | 29.8125 s | `(0, 118)` |

`prompts.txt` 是原始 30 条 prompt 的固定副本。每个 trajectory 文件也有
30 行，第 N 行与第 N 行 prompt 对齐。

## 构造方法

每条原始 20-pose action 被视为 outbound path，并按目标长度等比例扩展。
随后加入其逆序反向路径：

```text
outbound + inverse(reverse(outbound)) + one final action
```

例如：

```text
w*19,s*19,w*1
a*10,s*9,w*9,d*10,a*1
```

初始 pose 在倒数第二个 latent pose 被精确重访。之所以在 closure 后保留一个
action，是因为目标 latent pose 数均为偶数，而标准 action 字符串没有 no-op；
精确往返需要偶数个 action，加上 identity pose 后会得到奇数个 pose。

`manifest.csv` 和 `manifest.json` 记录每条轨迹的 prompt、源轨迹、闭环帧、
闭环误差和最大运动幅度。当前生成结果满足：

```text
closure translation error <= 7.3e-15
closure rotation error = 0 degree
```

这表示输入相机 pose 精确闭环，不代表生成视频一定视觉闭环。生成后仍需检查
minWM 是否正确遵循 camera action。

## 单组运行

以 10 秒轨迹为例：

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

CUDA_VISIBLE_DEVICES=0 \
DATA_PATH=Wan21/prompts/demos_loop_closure/prompts.txt \
TRAJECTORY_PATH=Wan21/prompts/demos_loop_closure/trajectories_10s.txt \
NUM_OUTPUT_FRAMES=40 \
MAX_PROMPTS=30 \
SEED=0 \
OUTPUT_FOLDER=./outputs/demo_loop_10s_seed0 \
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

其他时长分别设置：

```text
15s: TRAJECTORY_PATH=.../trajectories_15s.txt NUM_OUTPUT_FRAMES=60
20s: TRAJECTORY_PATH=.../trajectories_20s.txt NUM_OUTPUT_FRAMES=80
30s: TRAJECTORY_PATH=.../trajectories_30s.txt NUM_OUTPUT_FRAMES=120
```

## 运行四种时长的 baseline

```bash
cd /pool/hdd/home/hhhjyz/research/minWM
conda activate ling

for spec in "10s 40" "15s 60" "20s 80" "30s 120"; do
  read -r label frames <<< "$spec"
  CUDA_VISIBLE_DEVICES=0 \
  RUN_ROOT="./outputs/demo_loop_${label}" \
  DATA_PATH=Wan21/prompts/demos_loop_closure/prompts.txt \
  TRAJECTORY_PATH="Wan21/prompts/demos_loop_closure/trajectories_${label}.txt" \
  NUM_OUTPUT_FRAMES="$frames" \
  MAX_PROMPTS=30 \
  SEEDS="0" \
  CASES="baseline" \
  bash Wan21/scripts/inference/run_string_camera_experiments.sh
done
```

## 重新生成与验证

```bash
PYTHONPATH="$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}" \
python Wan21/scripts/data_preprocessing/build_string_loop_trajectories.py
```

构造器会在写出文件前检查 pose 数量、闭环平移误差和闭环旋转误差。
