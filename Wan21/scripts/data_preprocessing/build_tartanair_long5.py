#!/usr/bin/env python3
"""Build a small long-video camera-control benchmark for minWM.

This starter benchmark is based on TartanAir-style environments, but it does
not download raw TartanAir RGB/pose files. It creates prompt and trajectory
text files that are immediately compatible with Wan21/wan_inference.py.
"""

import argparse
import json
from pathlib import Path


SAMPLES = [
    {
        "id": "tartanair_abandoned_factory",
        "source_env": "TartanAir style: abandoned factory",
        "motion_ops": ["w", "w", "j", "w", "l"],
        "prompt": (
            "A cinematic first-person walkthrough inside an abandoned factory, "
            "with rusted steel beams, cracked concrete floors, scattered machinery, "
            "dusty shafts of light, and a slow exploratory camera moving through "
            "the industrial interior."
        ),
    },
    {
        "id": "tartanair_neighborhood",
        "source_env": "TartanAir style: neighborhood",
        "motion_ops": ["w", "d", "w", "a", "w"],
        "prompt": (
            "A quiet suburban neighborhood street under clear daylight, lined with "
            "small houses, trees, parked cars, fences, and sidewalks, viewed through "
            "a smooth forward-moving camera that gently shifts sideways while keeping "
            "long-term scene consistency."
        ),
    },
    {
        "id": "tartanair_oldtown",
        "source_env": "TartanAir style: old town",
        "motion_ops": ["j", "w", "w", "l", "w"],
        "prompt": (
            "A narrow old-town stone street with warm sunlight, textured walls, arched "
            "windows, hanging signs, and distant alleys, captured as a slow camera "
            "turns into the street and continues forward through the historic scene."
        ),
    },
    {
        "id": "tartanair_hospital",
        "source_env": "TartanAir style: hospital",
        "motion_ops": ["w", "i", "w", "k", "w"],
        "prompt": (
            "An empty hospital corridor with glossy floors, pale walls, overhead lights, "
            "medical carts, doors, and signs, filmed by a steady first-person camera "
            "that moves forward while slightly pitching up and down."
        ),
    },
    {
        "id": "tartanair_seaside",
        "source_env": "TartanAir style: seaside / coastal town",
        "motion_ops": ["w", "l", "w", "j", "w"],
        "prompt": (
            "A bright coastal walkway beside blue water, with stone railings, distant "
            "buildings, boats, and sunlit clouds, shown through a long smooth camera "
            "move that turns gently while preserving the layout of the scene."
        ),
    },
]


def split_counts(total_steps: int, parts: int) -> list[int]:
    if total_steps < parts:
        raise ValueError(
            f"num_frames is too small: need at least {parts + 1} frames, got {total_steps + 1}"
        )
    base = total_steps // parts
    rem = total_steps % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def make_trajectory(motion_ops: list[str], num_frames: int) -> str:
    total_steps = num_frames - 1
    counts = split_counts(total_steps, len(motion_ops))
    return ",".join(f"{op}*{count}" for op, count in zip(motion_ops, counts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Wan21/prompts/tartanair_long5"),
        help="Output directory relative to the minWM root when run from minWM.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=80,
        help="Latent/video frames for minWM generation. Use a multiple of 4 for causal camera.",
    )
    args = parser.parse_args()

    if args.num_frames <= 1:
        raise ValueError("--num-frames must be > 1")
    if args.num_frames % 4 != 0:
        raise ValueError("--num-frames should be a multiple of 4 for minWM causal camera")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = args.output_dir / "prompts.txt"
    trajectories_path = args.output_dir / "trajectories.txt"
    manifest_path = args.output_dir / "manifest.json"
    readme_path = args.output_dir / "README.md"

    prompts = []
    trajectories = []
    manifest_samples = []
    for sample in SAMPLES:
        traj = make_trajectory(sample["motion_ops"], args.num_frames)
        prompts.append(sample["prompt"])
        trajectories.append(traj)
        manifest_samples.append(
            {
                "id": sample["id"],
                "source_env": sample["source_env"],
                "prompt": sample["prompt"],
                "trajectory": traj,
                "num_frames": args.num_frames,
                "trajectory_steps": args.num_frames - 1,
            }
        )

    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    trajectories_path.write_text("\n".join(trajectories) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "name": "tartanair_long5",
                "source_dataset": "TartanAir",
                "source_url": "https://tartanair.org/",
                "status": (
                    "Starter benchmark using TartanAir-style scene categories and "
                    "minWM trajectory strings. Raw TartanAir RGB frames and GT poses "
                    "are not downloaded in this version."
                ),
                "num_samples": len(manifest_samples),
                "num_frames": args.num_frames,
                "format": {
                    "prompts": str(prompts_path),
                    "trajectories": str(trajectories_path),
                    "trajectory_convention": (
                        "minWM camera_trajectory.py string format; total operation "
                        "count equals num_frames - 1 because the first pose is identity."
                    ),
                },
                "samples": manifest_samples,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        f"""# TartanAir Long5 小规模长视频测试集

## 用途

这个目录提供 5 条适合 minWM 长视频推理实验的 prompt 和 camera trajectory，用于研究 sink frame、KV cache 和后续 FOV/WorldKV retrieval 对长视频生成的影响。

## 当前版本

这是 starter benchmark：使用 TartanAir 风格的场景类型和 minWM 原生 trajectory 字符串，不下载原始 TartanAir RGB 或 ground-truth pose。这样可以先在当前 minWM 推理接口上快速跑通实验。

## 文件

- `prompts.txt`：5 条长视频 prompt，每行一个样本。
- `trajectories.txt`：5 条 camera trajectory，每行对应 `prompts.txt` 的同一行。
- `manifest.json`：样本元信息、轨迹长度和来源说明。

默认每条 trajectory 对应 `NUM_OUTPUT_FRAMES={args.num_frames}`。minWM 的 trajectory 规则是首帧为 identity，因此每条 trajectory 的操作步数总和为 `{args.num_frames} - 1 = {args.num_frames - 1}`。

## 推理示例

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

DATA_PATH={prompts_path} \\
TRAJECTORY_PATH={trajectories_path} \\
NUM_OUTPUT_FRAMES={args.num_frames} \\
MAX_PROMPTS=5 \\
bash Wan21/scripts/inference/run_smoke_causal_camera.sh
```
""",
        encoding="utf-8",
    )

    print(f"Wrote {prompts_path}")
    print(f"Wrote {trajectories_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
