#!/usr/bin/env python3
"""Build prompt-aligned action-string trajectories with a late loop closure."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WAN_ROOT = PROJECT_ROOT / "Wan21"
sys.path.insert(0, str(WAN_ROOT))

from wan_utils.camera_trajectory import parse_trajectory  # noqa: E402


INVERSE_ACTION = {
    "w": "s",
    "s": "w",
    "a": "d",
    "d": "a",
    "u": "dn",
    "dn": "u",
    "j": "l",
    "l": "j",
    "i": "k",
    "k": "i",
}

DURATIONS = {
    "10s": 40,
    "15s": 60,
    "20s": 80,
    "30s": 120,
}


def parse_segments(value: str) -> list[tuple[str, int]]:
    segments = []
    for raw in value.strip().split(","):
        match = re.fullmatch(r"([a-z]+)\*(\d+)", raw.strip())
        if match is None:
            raise ValueError(f"Invalid trajectory segment: {raw!r}")
        action, count = match.group(1), int(match.group(2))
        if action not in INVERSE_ACTION or count <= 0:
            raise ValueError(f"Unsupported trajectory segment: {raw!r}")
        segments.append((action, count))
    return segments


def allocate_counts(segments: list[tuple[str, int]], target_steps: int) -> list[tuple[str, int]]:
    source_steps = sum(count for _, count in segments)
    exact = [count * target_steps / source_steps for _, count in segments]
    counts = [max(1, int(math.floor(value))) for value in exact]

    while sum(counts) < target_steps:
        order = sorted(
            range(len(counts)),
            key=lambda index: (exact[index] - math.floor(exact[index]), -index),
            reverse=True,
        )
        for index in order:
            if sum(counts) >= target_steps:
                break
            counts[index] += 1

    while sum(counts) > target_steps:
        order = sorted(
            range(len(counts)),
            key=lambda index: (exact[index] - math.floor(exact[index]), index),
        )
        changed = False
        for index in order:
            if sum(counts) <= target_steps:
                break
            if counts[index] > 1:
                counts[index] -= 1
                changed = True
        if not changed:
            raise ValueError(f"Cannot allocate {target_steps} steps across {segments}")

    return [(action, count) for (action, _), count in zip(segments, counts)]


def format_segments(segments: list[tuple[str, int]]) -> str:
    return ",".join(f"{action}*{count}" for action, count in segments if count > 0)


def build_loop_trajectory(source: str, latent_frames: int) -> tuple[str, int]:
    if latent_frames <= 2 or latent_frames % 2 != 0:
        raise ValueError("Loop trajectories require an even latent-frame count greater than 2")

    total_actions = latent_frames - 1
    outbound_steps = (total_actions - 1) // 2
    outbound = allocate_counts(parse_segments(source), outbound_steps)
    inbound = [(INVERSE_ACTION[action], count) for action, count in reversed(outbound)]
    tail = [(outbound[0][0], 1)]
    trajectory = format_segments([*outbound, *inbound, *tail])
    return trajectory, 2 * outbound_steps


def rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    return math.degrees(math.acos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0)))


def validate_trajectory(trajectory: str, latent_frames: int, closure_frame: int) -> dict[str, float]:
    viewmats = parse_trajectory(trajectory)
    if len(viewmats) != latent_frames:
        raise ValueError(f"Expected {latent_frames} poses, got {len(viewmats)}: {trajectory}")

    c2w = np.linalg.inv(viewmats.astype(np.float64))
    first_inv = np.linalg.inv(c2w[0])
    closure_rel = first_inv @ c2w[closure_frame]
    closure_translation = float(np.linalg.norm(closure_rel[:3, 3]))
    closure_rotation = rotation_angle_deg(closure_rel[:3, :3])

    positions = c2w[:, :3, 3]
    max_translation = float(np.linalg.norm(positions - positions[:1], axis=1).max())
    max_rotation = max(rotation_angle_deg(first_inv[:3, :3] @ pose[:3, :3]) for pose in c2w)

    if closure_translation > 1e-5 or closure_rotation > 1e-4:
        raise ValueError(
            f"Loop closure failed: translation={closure_translation}, rotation={closure_rotation}"
        )
    return {
        "closure_translation_error": closure_translation,
        "closure_rotation_error_deg": closure_rotation,
        "max_translation_from_start": max_translation,
        "max_rotation_from_start_deg": max_rotation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, default=WAN_ROOT / "prompts/demos.txt")
    parser.add_argument("--source-trajectories", type=Path, default=WAN_ROOT / "prompts/trajectories.txt")
    parser.add_argument("--output-dir", type=Path, default=WAN_ROOT / "prompts/demos_loop_closure")
    parser.add_argument("--fps", type=float, default=16.0)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = [line.strip() for line in args.source_trajectories.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != 30 or len(sources) != 30:
        raise ValueError(f"Expected 30 prompts and 30 trajectories, got {len(prompts)} and {len(sources)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")

    manifest = {
        "name": "demos_loop_closure",
        "source_prompts": str(args.prompts),
        "source_trajectories": str(args.source_trajectories),
        "num_prompts": len(prompts),
        "fps": args.fps,
        "time_mapping": "decoded_frames = 1 + 4 * (latent_frames - 1)",
        "closure_policy": "frame 0 exactly revisited at latent frame T-2; one final action follows",
        "durations": {},
    }
    csv_rows = []

    for label, latent_frames in DURATIONS.items():
        trajectory_path = args.output_dir / f"trajectories_{label}.txt"
        trajectories = []
        samples = []
        decoded_frames = 1 + 4 * (latent_frames - 1)

        for prompt_index, (prompt, source) in enumerate(zip(prompts, sources)):
            trajectory, closure_frame = build_loop_trajectory(source, latent_frames)
            metrics = validate_trajectory(trajectory, latent_frames, closure_frame)
            trajectories.append(trajectory)
            sample = {
                "prompt_index": prompt_index,
                "prompt": prompt,
                "source_trajectory": source,
                "trajectory": trajectory,
                "closure_pair_latent": [0, closure_frame],
                "closure_pair_decoded": [0, 4 * closure_frame],
                **metrics,
            }
            samples.append(sample)
            csv_rows.append({
                "duration_label": label,
                "target_seconds": int(label[:-1]),
                "actual_seconds": decoded_frames / args.fps,
                "latent_frames": latent_frames,
                "decoded_frames": decoded_frames,
                **sample,
                "closure_pair_latent": f"0:{closure_frame}",
                "closure_pair_decoded": f"0:{4 * closure_frame}",
            })

        trajectory_path.write_text("\n".join(trajectories) + "\n", encoding="utf-8")
        manifest["durations"][label] = {
            "trajectory_file": str(trajectory_path),
            "latent_frames": latent_frames,
            "decoded_frames": decoded_frames,
            "actual_seconds": decoded_frames / args.fps,
            "samples": samples,
        }

    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output_dir / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(csv_rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"Wrote {args.output_dir / 'prompts.txt'}")
    for label in DURATIONS:
        print(f"Wrote {args.output_dir / f'trajectories_{label}.txt'}")
    print(f"Wrote {args.output_dir / 'manifest.json'}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
