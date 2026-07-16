#!/usr/bin/env python3
"""Evaluate minWM outputs on the ViewBench small split.

The first version focuses on loop-closure consistency inside each generated
video: compare frame pairs listed in the small-split manifest, defaulting to the
first and last frames.  This is more meaningful for generative minWM experiments
than pixel-matching against UE5 ground truth scenes that the model was not
conditioned to reconstruct exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.io import read_video


def _to_float(value: Any):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_timing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(dict(row))
    return rows


def _read_video_rgb(path: Path) -> torch.Tensor:
    video, _, _ = read_video(str(path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Cannot read video frames from {path}")
    # (T,H,W,C) uint8 -> (T,C,H,W) float in [0,1]
    return video.permute(0, 3, 1, 2).float() / 255.0


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def _ssim_global(a: torch.Tensor, b: torch.Tensor) -> float:
    # Lightweight dependency-free SSIM over RGB tensors.  It is not windowed
    # SSIM, but is stable enough for small ablations and quick smoke checks.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    mu_a = a.mean(dim=1)
    mu_b = b.mean(dim=1)
    var_a = a.var(dim=1, unbiased=False)
    var_b = b.var(dim=1, unbiased=False)
    cov = ((a - mu_a[:, None]) * (b - mu_b[:, None])).mean(dim=1)
    score = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return float(score.mean().item())


def _maybe_lpips(device: str):
    try:
        import lpips  # type: ignore

        model = lpips.LPIPS(net="alex").to(device)
        model.eval()
        return model, None
    except Exception as exc:
        return None, repr(exc)


@torch.no_grad()
def _lpips_score(model, a: torch.Tensor, b: torch.Tensor, device: str) -> float | None:
    if model is None:
        return None
    a = a.unsqueeze(0).to(device) * 2.0 - 1.0
    b = b.unsqueeze(0).to(device) * 2.0 - 1.0
    return float(model(a, b).mean().item())


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pair-radius",
        type=int,
        default=0,
        help="Average frames within +/- radius around each loop-closure endpoint.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    timing_rows = _load_timing(args.timing_csv)
    generated = [row for row in timing_rows if row.get("status") == "generated"]
    timing_by_prompt_index = {int(row["prompt_index"]): row for row in generated}

    lpips_model, lpips_error = _maybe_lpips(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = args.output_dir / "viewbench_metrics.csv"
    metrics_json = args.output_dir / "viewbench_metrics.json"

    metric_rows = []
    for prompt_index, sample in enumerate(samples):
        timing = timing_by_prompt_index.get(prompt_index)
        if timing is None:
            continue
        video_path = Path(timing["output_path"])
        if not video_path.exists():
            continue
        video = _read_video_rgb(video_path)
        num_video_frames = int(video.shape[0])
        pairs = sample.get("loop_closure_pairs") or [[0, num_video_frames - 1]]

        pair_psnr = []
        pair_ssim = []
        pair_lpips = []
        used_pairs = []
        for start, end in pairs:
            start = max(0, min(int(start), num_video_frames - 1))
            end = max(0, min(int(end), num_video_frames - 1))
            radius = max(0, int(args.pair_radius))
            start_slice = video[max(0, start - radius): min(num_video_frames, start + radius + 1)].mean(dim=0)
            end_slice = video[max(0, end - radius): min(num_video_frames, end + radius + 1)].mean(dim=0)
            pair_psnr.append(_psnr(start_slice, end_slice))
            pair_ssim.append(_ssim_global(start_slice, end_slice))
            pair_lpips.append(_lpips_score(lpips_model, start_slice, end_slice, args.device))
            used_pairs.append([start, end])

        row = {
            "prompt_index": prompt_index,
            "bucket": sample.get("bucket", ""),
            "sequence_id": sample.get("sequence_id", ""),
            "output_path": str(video_path),
            "num_video_frames": num_video_frames,
            "loop_closure_pairs": json.dumps(used_pairs),
            "psnr": _mean(pair_psnr),
            "ssim": _mean(pair_ssim),
            "lpips": _mean(pair_lpips),
            "last_chunk_fps": _to_float(timing.get("last_chunk_fps")),
            "peak_vram_allocated_gb": _to_float(timing.get("peak_vram_allocated_gb")),
            "peak_vram_reserved_gb": _to_float(timing.get("peak_vram_reserved_gb")),
            "retrieval_latency_seconds": _to_float(timing.get("retrieval_latency_seconds")),
        }
        metric_rows.append(row)

    fieldnames = [
        "prompt_index",
        "bucket",
        "sequence_id",
        "output_path",
        "num_video_frames",
        "loop_closure_pairs",
        "psnr",
        "ssim",
        "lpips",
        "last_chunk_fps",
        "peak_vram_allocated_gb",
        "peak_vram_reserved_gb",
        "retrieval_latency_seconds",
    ]
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {
        "manifest": str(args.manifest),
        "timing_csv": str(args.timing_csv),
        "num_evaluated": len(metric_rows),
        "metrics": {
            "avg_psnr": _mean([row["psnr"] for row in metric_rows]),
            "avg_ssim": _mean([row["ssim"] for row in metric_rows]),
            "avg_lpips": _mean([row["lpips"] for row in metric_rows]),
            "avg_last_chunk_fps": _mean([row["last_chunk_fps"] for row in metric_rows]),
            "max_peak_vram_allocated_gb": max(
                [row["peak_vram_allocated_gb"] for row in metric_rows if row["peak_vram_allocated_gb"] is not None],
                default=None,
            ),
            "avg_retrieval_latency_seconds": _mean(
                [row["retrieval_latency_seconds"] for row in metric_rows]
            ),
            "fid": None,
        },
        "notes": {
            "lpips": "computed" if lpips_model is not None else f"skipped: {lpips_error}",
            "psnr_ssim": "computed on generated loop-closure frame pairs, not against UE5 ground truth",
            "fid": "not computed in the lightweight evaluator; add a reference distribution before using FID",
        },
        "videos": metric_rows,
    }
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {metrics_json}")


if __name__ == "__main__":
    main()
