#!/usr/bin/env python3
"""Build a 12-sequence ViewBench small split for minWM experiments.

The script expects ViewBench shards to have been extracted as described by the
dataset README.  It does not download the large ``tar.zst`` archives.  Output is
directly consumable by ``Wan21/wan_inference.py`` via:

  --data_path prompts.txt --trajectory_pose_path trajectory_pose_paths.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WAN21_ROOT = Path(__file__).resolve().parents[2]
if str(WAN21_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN21_ROOT))

from wan_utils.viewbench_pose import (
    c2w_ue_to_viewmats,
    load_viewbench_json_c2w,
    make_intrinsics,
    save_pose_npz,
)


@dataclass
class Candidate:
    group: str
    sequence_id: str
    json_path: Path
    frame_dir: Path | None
    num_frames: int
    translation_m: float
    yaw_range_deg: float
    pitch_range_deg: float
    roll_range_deg: float
    varied_axes: int


def _rotation_to_euler_ranges(c2w: np.ndarray) -> tuple[float, float, float]:
    rotations = []
    for mat in c2w:
        r = mat[:3, :3]
        # For ViewBench convention R = Rz(yaw) * Ry(pitch) * Rx(roll).
        pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
        roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
        yaw = math.atan2(float(r[1, 0]), float(r[0, 0]))
        rotations.append([pitch, roll, yaw])
    arr = np.unwrap(np.asarray(rotations), axis=0)
    ranges = np.degrees(arr.max(axis=0) - arr.min(axis=0))
    pitch_range, roll_range, yaw_range = ranges.tolist()
    return yaw_range, pitch_range, roll_range


def _find_dataset_root(root: Path) -> Path:
    candidates = [
        root,
        root / "ViewBench4Training",
        root / "ViewBench4Training" / "ViewBench4Training",
    ]
    for candidate in candidates:
        if (candidate / "pure_rotation").exists() or (candidate / "rotation_translation").exists():
            return candidate
    raise FileNotFoundError(
        f"Cannot find extracted ViewBench directories under {root}. Expected "
        "pure_rotation/ and rotation_translation/."
    )


def _scan_group(dataset_root: Path, group: str, limit: int | None = None) -> list[Candidate]:
    json_dir = dataset_root / group / "jsons"
    frame_root = dataset_root / group / "frames"
    if not json_dir.exists():
        return []
    candidates = []
    for json_path in sorted(json_dir.glob("*.json")):
        c2w = load_viewbench_json_c2w(json_path)
        translation_m = float(np.linalg.norm((c2w[-1, :3, 3] - c2w[0, :3, 3]) * 0.01))
        yaw, pitch, roll = _rotation_to_euler_ranges(c2w)
        varied_axes = sum(v >= 8.0 for v in (yaw, pitch, roll))
        sequence_id = json_path.stem
        frame_dir = frame_root / sequence_id
        candidates.append(
            Candidate(
                group=group,
                sequence_id=sequence_id,
                json_path=json_path,
                frame_dir=frame_dir if frame_dir.exists() else None,
                num_frames=len(c2w),
                translation_m=translation_m,
                yaw_range_deg=float(yaw),
                pitch_range_deg=float(pitch),
                roll_range_deg=float(roll),
                varied_axes=varied_axes,
            )
        )
        if limit is not None and len(candidates) >= limit:
            break
    return candidates


def _round_robin_by_scene(candidates: list[Candidate], count: int) -> list[Candidate]:
    selected = []
    seen_scene = set()
    for item in candidates:
        scene = item.sequence_id.rsplit("_", 1)[0]
        if scene in seen_scene:
            continue
        selected.append(item)
        seen_scene.add(scene)
        if len(selected) >= count:
            return selected
    for item in candidates:
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            return selected
    return selected


def _select(candidates: list[Candidate], group_name: str, count: int) -> list[Candidate]:
    if group_name == "pure_rotation":
        pool = [c for c in candidates if c.group == "pure_rotation" and c.varied_axes <= 1]
        pool = sorted(pool, key=lambda c: (-max(c.yaw_range_deg, c.pitch_range_deg, c.roll_range_deg), c.sequence_id))
    elif group_name == "multi_axis_rotation":
        pool = [c for c in candidates if c.group == "pure_rotation" and c.varied_axes >= 2]
        pool = sorted(pool, key=lambda c: (-c.varied_axes, -c.yaw_range_deg - c.pitch_range_deg - c.roll_range_deg, c.sequence_id))
    elif group_name == "rotation_translation":
        pool = [c for c in candidates if c.group == "rotation_translation"]
        pool = sorted(pool, key=lambda c: (-c.translation_m, -c.varied_axes, c.sequence_id))
    else:
        raise ValueError(group_name)

    if len(pool) < count and group_name == "multi_axis_rotation":
        fallback = [c for c in candidates if c.group == "pure_rotation" and c not in pool]
        pool.extend(sorted(fallback, key=lambda c: (-c.varied_axes, c.sequence_id)))
    if len(pool) < count:
        raise RuntimeError(f"Only found {len(pool)} candidates for {group_name}, need {count}")
    return _round_robin_by_scene(pool, count)


def _prompt(item: Candidate, bucket: str) -> str:
    scene = item.sequence_id.rsplit("_", 1)[0].replace("_", " ").lower()
    if bucket == "pure_rotation":
        motion = "a stationary camera rotates away and returns to a previously seen viewpoint"
    elif bucket == "multi_axis_rotation":
        motion = "a stationary camera performs multi-axis rotation with a loop closure"
    else:
        motion = "a moving camera combines rotation and translation before returning near an earlier view"
    return (
        f"A photorealistic long-horizon first-person video in a {scene} environment, "
        f"captured as {motion}, with stable scene layout and consistent appearance."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewbench-root", type=Path, required=True, help="Extracted ViewBench root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Wan21/prompts/viewbench_small12"),
        help="Output directory under the minWM repo.",
    )
    parser.add_argument("--samples-per-bucket", type=int, default=4)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=360,
        help="Truncate pose sequences to at most this many frames. <=0 keeps native length.",
    )
    parser.add_argument(
        "--scan-limit-per-group",
        type=int,
        default=0,
        help="Debug option: only scan the first N JSON files in each group. 0 scans all.",
    )
    parser.add_argument("--translation-scale", type=float, default=0.01, help="ViewBench cm-to-model scale.")
    parser.add_argument("--no-recenter", action="store_true", help="Do not recenter first pose to identity.")
    parser.add_argument(
        "--no-axis-conversion",
        action="store_true",
        help="Keep ViewBench UE camera axes instead of converting to OpenCV-like camera axes.",
    )
    args = parser.parse_args()

    dataset_root = _find_dataset_root(args.viewbench_root)
    scan_limit = args.scan_limit_per_group if args.scan_limit_per_group > 0 else None
    candidates = []
    candidates.extend(_scan_group(dataset_root, "pure_rotation", scan_limit))
    candidates.extend(_scan_group(dataset_root, "rotation_translation", scan_limit))
    if not candidates:
        raise RuntimeError(f"No ViewBench candidates found in {dataset_root}")

    buckets = {
        "pure_rotation": _select(candidates, "pure_rotation", args.samples_per_bucket),
        "multi_axis_rotation": _select(candidates, "multi_axis_rotation", args.samples_per_bucket),
        "rotation_translation": _select(candidates, "rotation_translation", args.samples_per_bucket),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = args.output_dir / "poses"
    prompts_path = args.output_dir / "prompts.txt"
    pose_paths_path = args.output_dir / "trajectory_pose_paths.txt"
    manifest_path = args.output_dir / "manifest.json"
    readme_path = args.output_dir / "README.md"

    prompts = []
    pose_paths = []
    samples = []
    for bucket, items in buckets.items():
        for item in items:
            c2w = load_viewbench_json_c2w(item.json_path)
            if args.max_frames > 0:
                c2w = c2w[: args.max_frames]
            viewmats = c2w_ue_to_viewmats(
                c2w,
                translation_scale=args.translation_scale,
                convert_camera_axes=not args.no_axis_conversion,
                recenter=not args.no_recenter,
            )
            Ks = make_intrinsics(len(viewmats))
            pose_path = pose_dir / f"{bucket}_{item.sequence_id}.npz"
            save_pose_npz(pose_path, viewmats, Ks)

            prompt = _prompt(item, bucket)
            prompts.append(prompt)
            pose_paths.append(str(pose_path))
            samples.append(
                {
                    "bucket": bucket,
                    "sequence_id": item.sequence_id,
                    "prompt": prompt,
                    "pose_npz": str(pose_path),
                    "source_json": str(item.json_path),
                    "source_frames": str(item.frame_dir) if item.frame_dir else None,
                    "num_frames": int(len(viewmats)),
                    "loop_closure_pairs": [[0, int(len(viewmats) - 1)]],
                    "motion_summary": {
                        "translation_m": item.translation_m,
                        "yaw_range_deg": item.yaw_range_deg,
                        "pitch_range_deg": item.pitch_range_deg,
                        "roll_range_deg": item.roll_range_deg,
                        "varied_axes": item.varied_axes,
                    },
                }
            )

    prompts_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    pose_paths_path.write_text("\n".join(pose_paths) + "\n", encoding="utf-8")
    manifest = {
        "name": "viewbench_small12",
        "source_dataset": "JEdward/viewbench-dataset",
        "source_url": "https://huggingface.co/datasets/JEdward/viewbench-dataset",
        "dataset_root": str(dataset_root),
        "num_samples": len(samples),
        "buckets": {key: len(value) for key, value in buckets.items()},
        "format": {
            "prompts": str(prompts_path),
            "trajectory_pose_paths": str(pose_paths_path),
            "pose_npz": {"viewmats": "(T,4,4)", "Ks": "(T,3,3)"},
            "loop_closure_pairs": "default first/last frame pair; replace with overlap metadata later if needed",
        },
        "conversion": {
            "translation_scale": args.translation_scale,
            "axis_conversion": not args.no_axis_conversion,
            "recenter_first_pose": not args.no_recenter,
            "max_frames": args.max_frames,
        },
        "samples": samples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    readme_path.write_text(
        f"""# ViewBench Small12 for minWM

这个目录由 `Wan21/scripts/data_preprocessing/build_viewbench_small.py` 生成，用于 minWM 的
long-horizon camera-control / loop-closure 实验。

## 文件

- `prompts.txt`：12 条 prompt。
- `trajectory_pose_paths.txt`：每行一个 pose `.npz`，与 prompt 对齐。
- `poses/*.npz`：包含 `viewmats: (T,4,4)` 和 `Ks: (T,3,3)`。
- `manifest.json`：样本分组、源文件、轨迹摘要和 loop-closure pair。

## 默认分组

- 4 条纯旋转。
- 4 条多轴旋转。
- 4 条旋转+平移。

## 推理示例

```bash
cd /pool/hdd/home/hhhjyz/research/minWM

DATA_PATH={prompts_path} \\
TRAJECTORY_POSE_PATH={pose_paths_path} \\
NUM_OUTPUT_FRAMES=180 \\
MAX_PROMPTS=1 \\
bash Wan21/scripts/inference/run_profiled_smoke_causal_camera.sh
```

如果 `NUM_OUTPUT_FRAMES<=0`，推理会使用每条 pose `.npz` 的原生长度。不同样本长度可以共存，
但长视频推理会更慢，正式比较时建议按 180/240/360 分组运行。
""",
        encoding="utf-8",
    )

    print(f"Wrote {prompts_path}")
    print(f"Wrote {pose_paths_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
