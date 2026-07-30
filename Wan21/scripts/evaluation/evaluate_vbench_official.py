#!/usr/bin/env python3
"""Run official VBench/VBench-Long on minWM generated videos.

This is a thin adapter around the official VBench repository.  It does not
vendor VBench or install dependencies.  It prepares a flat video folder from
minWM's ``inference_times.csv``, invokes the official evaluator per dimension,
and aggregates the raw JSON outputs into CSV/JSON files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any


VBENCH_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

VBENCH_LONG_QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def _load_timing(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _safe_name(index: int, row: dict[str, Any]) -> str:
    source = Path(row.get("output_path", "")).stem
    suffix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source)
    return f"{int(row.get('prompt_index', index)):04d}_{suffix[:120]}.mp4"


def prepare_video_dir(timing_csv: Path, output_dir: Path, *, copy_videos: bool) -> tuple[Path, list[dict[str, Any]]]:
    rows = [
        row
        for row in _load_timing(timing_csv)
        if row.get("status") in {"generated", "skipped_exists"}
    ]
    video_dir = output_dir / "official_vbench_videos"
    if video_dir.exists():
        shutil.rmtree(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for index, row in enumerate(rows):
        src = Path(row.get("output_path", "")).resolve()
        if not src.exists():
            continue
        dst = video_dir / _safe_name(index, row)
        if copy_videos:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        manifest.append({
            "prompt_index": row.get("prompt_index", ""),
            "prompt": row.get("prompt", ""),
            "source_video": str(src),
            "vbench_video": str(dst.resolve()),
        })

    if not manifest:
        raise RuntimeError(f"No generated videos found in {timing_csv}")
    (output_dir / "official_vbench_video_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return video_dir, manifest


def run_dimension(
    *,
    vbench_root: Path,
    python_bin: str,
    videos_path: Path,
    output_path: Path,
    dimension: str,
    long: bool,
    master_port: int,
    load_ckpt_from_local: bool,
    env_overrides: dict[str, str],
    extra_args: list[str],
) -> None:
    if long:
        script = vbench_root / "vbench2_beta_long" / "eval_long.py"
        cmd = [
            python_bin,
            str(script),
            "--videos_path",
            str(videos_path),
            "--output_path",
            str(output_path),
            "--dimension",
            dimension,
            "--mode",
            "long_custom_input",
            "--dev_flag",
            *extra_args,
        ]
    else:
        script = vbench_root / "evaluate.py"
        cmd = [
            python_bin,
            str(script),
            "--videos_path",
            str(videos_path),
            "--dimension",
            dimension,
            "--mode",
            "custom_input",
            "--output_path",
            str(output_path),
            *extra_args,
        ]
        if load_ckpt_from_local:
            cmd.extend(["--load_ckpt_from_local", "True"])
    if not script.exists():
        raise FileNotFoundError(f"Cannot find official VBench entrypoint: {script}")

    env = os.environ.copy()
    env.update(env_overrides)
    env["MASTER_PORT"] = str(master_port)
    print("[official-vbench]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(vbench_root), env=env, check=True)


def _parse_score(value: Any):
    if isinstance(value, bool):
        return int(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_free_port(preferred: int) -> int:
    """Return preferred if free, otherwise ask the OS for an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", preferred))
        except OSError:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as fallback:
                fallback.bind(("", 0))
                return int(fallback.getsockname()[1])
        return preferred


def collect_results(eval_dir: Path, dimensions: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_video: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {}

    for result_file in sorted(eval_dir.glob("results_*_eval_results.json")):
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        for dimension in dimensions:
            if dimension not in payload:
                continue
            result = payload[dimension]
            if isinstance(result, list) and result:
                summary[dimension] = _parse_score(result[0])
                entries = result[1] if len(result) > 1 and isinstance(result[1], list) else []
            else:
                summary[dimension] = None
                entries = []
            for item in entries:
                video_path = str(Path(item.get("video_path", "")).resolve())
                row = per_video.setdefault(video_path, {"video_path": video_path})
                row[dimension] = _parse_score(item.get("video_results"))

    rows = [per_video[key] for key in sorted(per_video)]
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dimensions", nargs="*", default=None)
    parser.add_argument("--long", action="store_true", help="Use VBench-Long eval_long.py.")
    parser.add_argument("--copy-videos", action="store_true", help="Copy videos instead of symlinking.")
    parser.add_argument("--master-port", type=int, default=38600)
    parser.add_argument(
        "--load-ckpt-from-local",
        action="store_true",
        help="Pass --load_ckpt_from_local True to official VBench.",
    )
    parser.add_argument(
        "--vbench-cache-dir",
        type=Path,
        default=None,
        help="Directory used as VBENCH_CACHE_DIR for official VBench checkpoints.",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        default=None,
        help="Optional TORCH_HOME used by torch hub downloads/cache.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=None,
        help="Optional HF_HOME used by Hugging Face downloads/cache.",
    )
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw args appended to official VBench command.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = args.dimensions or (VBENCH_LONG_QUALITY_DIMS if args.long else VBENCH_QUALITY_DIMS)
    video_dir, manifest = prepare_video_dir(args.timing_csv, args.output_dir, copy_videos=args.copy_videos)
    raw_dir = args.output_dir / ("official_vbench_long_raw" if args.long else "official_vbench_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    env_overrides = {}
    if args.vbench_cache_dir is not None:
        env_overrides["VBENCH_CACHE_DIR"] = str(args.vbench_cache_dir.resolve())
    if args.torch_home is not None:
        env_overrides["TORCH_HOME"] = str(args.torch_home.resolve())
    if args.hf_home is not None:
        env_overrides["HF_HOME"] = str(args.hf_home.resolve())

    for offset, dimension in enumerate(dimensions):
        master_port = find_free_port(args.master_port + offset)
        run_dimension(
            vbench_root=args.vbench_root.resolve(),
            python_bin=args.python_bin,
            videos_path=video_dir.resolve(),
            output_path=raw_dir.resolve(),
            dimension=dimension,
            long=args.long,
            master_port=master_port,
            load_ckpt_from_local=bool(args.load_ckpt_from_local),
            env_overrides=env_overrides,
            extra_args=args.extra_arg,
        )

    rows, summary = collect_results(raw_dir, dimensions)
    csv_path = args.output_dir / ("official_vbench_long_metrics.csv" if args.long else "official_vbench_metrics.csv")
    fieldnames = ["video_path", *dimensions]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.output_dir / ("official_vbench_long_metrics.json" if args.long else "official_vbench_metrics.json")
    json_path.write_text(
        json.dumps(
            {
                "official_vbench": True,
                "long": bool(args.long),
                "vbench_root": str(args.vbench_root),
                "timing_csv": str(args.timing_csv),
                "dimensions": dimensions,
                "load_ckpt_from_local": bool(args.load_ckpt_from_local),
                "vbench_cache_dir": str(args.vbench_cache_dir) if args.vbench_cache_dir else None,
                "num_videos": len(manifest),
                "summary": summary,
                "videos": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
