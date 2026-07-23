#!/usr/bin/env python3
"""Lightweight VBench-style no-reference metrics for generated videos.

This script intentionally does not claim official VBench scores.  It evaluates
our own generated videos without requiring VBench prompts or heavyweight model
checkpoints, and writes metrics that are useful for internal ablations:

  - temporal frame difference / flicker / smoothness proxy
  - dynamic degree proxy from dense optical flow
  - sharpness, brightness, contrast, black/white frame ratios

Input is the minWM ``inference_times.csv`` produced by ``wan_inference.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _to_float(value: Any):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def _load_timing(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _read_video_rgb(path: Path, *, max_frames: int = 0, resize_width: int = 256) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = 1
    if max_frames > 0 and frame_count > max_frames:
        stride = max(1, frame_count // max_frames)

    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        index += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _flow_magnitude(prev_gray: np.ndarray, next_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=15,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(mag))


def evaluate_video(path: Path, *, max_frames: int, resize_width: int) -> dict[str, Any]:
    video = _read_video_rgb(path, max_frames=max_frames, resize_width=resize_width)
    num_frames = int(video.shape[0])
    gray = np.stack(
        [cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0 for frame in video]
    )

    brightness = gray.mean(axis=(1, 2))
    contrast = gray.std(axis=(1, 2))
    sharpness = []
    for frame_gray in gray:
        lap = cv2.Laplacian((frame_gray * 255).astype(np.uint8), cv2.CV_64F)
        sharpness.append(float(lap.var()))

    if num_frames >= 2:
        diffs = np.abs(video[1:] - video[:-1]).mean(axis=(1, 2, 3))
        gray_diffs = gray[1:] - gray[:-1]
        frame_diff_mean = float(diffs.mean())
        frame_diff_std = float(diffs.std())
        if num_frames >= 3:
            accel = np.abs(gray_diffs[1:] - gray_diffs[:-1]).mean(axis=(1, 2))
            temporal_flicker = float(accel.mean())
        else:
            temporal_flicker = 0.0

        flow_values = [
            _flow_magnitude((gray[i] * 255).astype(np.uint8), (gray[i + 1] * 255).astype(np.uint8))
            for i in range(num_frames - 1)
        ]
        optical_flow_mean = float(np.mean(flow_values))
        optical_flow_std = float(np.std(flow_values))
    else:
        frame_diff_mean = 0.0
        frame_diff_std = 0.0
        temporal_flicker = 0.0
        optical_flow_mean = 0.0
        optical_flow_std = 0.0

    smoothness_score = float(1.0 / (1.0 + temporal_flicker))
    dynamic_degree_proxy = optical_flow_mean
    black_frame_ratio = float(np.mean(brightness < 0.03))
    white_frame_ratio = float(np.mean(brightness > 0.97))

    return {
        "num_metric_frames": num_frames,
        "brightness_mean": float(brightness.mean()),
        "brightness_std": float(brightness.std()),
        "contrast_mean": float(contrast.mean()),
        "sharpness_laplacian_var": float(np.mean(sharpness)),
        "frame_difference_mean": frame_diff_mean,
        "frame_difference_std": frame_diff_std,
        "temporal_flicker_proxy": temporal_flicker,
        "motion_smoothness_proxy": smoothness_score,
        "dynamic_degree_proxy": dynamic_degree_proxy,
        "optical_flow_mean": optical_flow_mean,
        "optical_flow_std": optical_flow_std,
        "black_frame_ratio": black_frame_ratio,
        "white_frame_ratio": white_frame_ratio,
        "subject_consistency_proxy": float(1.0 / (1.0 + frame_diff_mean)),
        "background_consistency_proxy": float(1.0 / (1.0 + temporal_flicker)),
        "temporal_flickering_proxy_vbench": float(1.0 / (1.0 + temporal_flicker)),
        "motion_smoothness_proxy_vbench": smoothness_score,
        "dynamic_degree_proxy_vbench": dynamic_degree_proxy,
        "aesthetic_quality_proxy": float(np.clip((np.mean(sharpness) / 3000.0) * contrast.mean(), 0.0, 1.0)),
        "imaging_quality_proxy": float(
            np.clip(
                0.5 * min(np.mean(sharpness) / 3000.0, 1.0)
                + 0.3 * min(contrast.mean() / 0.25, 1.0)
                + 0.2 * (1.0 - black_frame_ratio - white_frame_ratio),
                0.0,
                1.0,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--resize-width", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _load_timing(args.timing_csv)
        if row.get("status") in {"generated", "skipped_exists"}
    ]

    metric_rows = []
    errors = []
    for row in rows:
        video_path = Path(row.get("output_path", ""))
        if not video_path.exists():
            errors.append({"output_path": str(video_path), "error": "missing video"})
            continue
        try:
            metrics = evaluate_video(
                video_path,
                max_frames=max(0, int(args.max_frames)),
                resize_width=max(0, int(args.resize_width)),
            )
        except Exception as exc:
            errors.append({"output_path": str(video_path), "error": repr(exc)})
            continue

        metric_rows.append({
            "sample_order": row.get("sample_order", ""),
            "prompt_index": row.get("prompt_index", ""),
            "prompt": row.get("prompt", ""),
            "output_path": str(video_path),
            "last_chunk_fps": _to_float(row.get("last_chunk_fps")),
            "generation_seconds": _to_float(row.get("generation_seconds")),
            **metrics,
        })

    fieldnames = [
        "sample_order",
        "prompt_index",
        "prompt",
        "output_path",
        "last_chunk_fps",
        "generation_seconds",
        "num_metric_frames",
        "brightness_mean",
        "brightness_std",
        "contrast_mean",
        "sharpness_laplacian_var",
        "frame_difference_mean",
        "frame_difference_std",
        "temporal_flicker_proxy",
        "motion_smoothness_proxy",
        "dynamic_degree_proxy",
        "optical_flow_mean",
        "optical_flow_std",
        "black_frame_ratio",
        "white_frame_ratio",
        "subject_consistency_proxy",
        "background_consistency_proxy",
        "temporal_flickering_proxy_vbench",
        "motion_smoothness_proxy_vbench",
        "dynamic_degree_proxy_vbench",
        "aesthetic_quality_proxy",
        "imaging_quality_proxy",
    ]
    metrics_csv = args.output_dir / "vbench_style_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "timing_csv": str(args.timing_csv),
        "num_evaluated": len(metric_rows),
        "num_errors": len(errors),
        "metrics": {
            key: _mean([row.get(key) for row in metric_rows])
            for key in fieldnames
            if key not in {"sample_order", "prompt_index", "prompt", "output_path"}
        },
        "notes": {
            "official_vbench": False,
            "prompt_usage": "Uses the project's own prompts/videos; no VBench prompt set is required.",
            "text_alignment": "Not computed because clip/open_clip is not installed in the current environment.",
            "dynamic_degree_proxy": "Dense optical-flow magnitude averaged over sampled adjacent frames.",
            "temporal_flicker_proxy": "Mean second-order grayscale frame difference; lower is better.",
            "motion_smoothness_proxy": "1 / (1 + temporal_flicker_proxy); higher is better.",
            "vbench_named_proxies": "Columns ending in _proxy or _proxy_vbench are lightweight internal proxies aligned to VBench dimension names, not official VBench scores.",
        },
        "errors": errors,
        "videos": metric_rows,
    }
    metrics_json = args.output_dir / "vbench_style_metrics.json"
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {metrics_json}")


if __name__ == "__main__":
    main()
