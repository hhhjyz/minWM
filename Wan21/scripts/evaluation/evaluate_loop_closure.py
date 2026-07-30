#!/usr/bin/env python3
"""Evaluate visual loop closure inside generated minWM videos.

The demo loop trajectories travel along an outbound path and then follow the
exact inverse path back to the initial pose.  This makes the outbound half of
each generated video an internal reference for the revisit half, even though
there is no external ground-truth video.

The main ``psnr``, ``ssim``, and ``lpips`` metrics follow MAG-Bench: every
revisit frame first matches to its LPIPS-nearest outbound frame, and PSNR/SSIM
are computed on those same matched pairs.  This tolerates different traversal
speeds, while the accompanying match diagnostics expose duplicated or
temporally invalid correspondences.

The exact closure and decoded-frame indices come from
``prompts/demos_loop_closure/manifest.json``.  The turnaround frame is excluded
from both path segments so that a trivially identical shared boundary frame
does not inflate the metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METRIC_FIELDS = [
    "psnr",
    "ssim",
    "lpips",
    "closure_psnr",
    "closure_ssim",
    "closure_lpips",
    "mag_psnr",
    "mag_ssim",
    "mag_lpips",
    "match_unique_ratio",
    "match_temporal_mae_normalized",
    "match_reverse_violation_ratio",
]


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[Any]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(valid) / len(valid)) if valid else None


def _load_timing(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _resolve_video_path(raw_path: str, timing_csv: Path) -> Path:
    raw = Path(raw_path)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(timing_csv.parent / raw)
    candidates.append(timing_csv.parent / raw.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return raw


def _duration_label(
    manifest: dict[str, Any], row: dict[str, Any], requested_label: str | None
) -> str:
    durations = manifest.get("durations", {})
    if requested_label:
        if requested_label not in durations:
            raise ValueError(
                f"Unknown duration label {requested_label!r}; expected one of {sorted(durations)}"
            )
        return requested_label

    try:
        latent_frames = int(row.get("num_output_frames", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError("Cannot infer duration: timing row has no num_output_frames") from exc
    matches = [
        label
        for label, spec in durations.items()
        if int(spec.get("latent_frames", -1)) == latent_frames
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Cannot uniquely map num_output_frames={latent_frames} to manifest duration: {matches}"
        )
    return matches[0]


def _manifest_sample(
    manifest: dict[str, Any], duration_label: str, prompt_index: int
) -> dict[str, Any]:
    samples = manifest["durations"][duration_label].get("samples", [])
    for sample in samples:
        if int(sample.get("prompt_index", -1)) == prompt_index:
            return sample
    raise ValueError(
        f"No manifest sample for duration={duration_label!r}, prompt_index={prompt_index}"
    )


def _uniform_indices(length: int, maximum: int) -> list[int]:
    if length <= 0:
        return []
    if maximum <= 0 or length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [length // 2]
    denominator = maximum - 1
    # Integer round-to-nearest keeps the first and last samples and avoids
    # platform-dependent floating-point rounding.
    return [
        (index * (length - 1) + denominator // 2) // denominator
        for index in range(maximum)
    ]


def _read_selected_video_frames(
    path: Path, selected_indices: list[int], resize_width: int, cv2, np
) -> tuple[dict[int, Any], int]:
    wanted = set(selected_indices)
    if not wanted:
        raise ValueError("No video frames selected")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: dict[int, Any] = {}
    last_needed = max(wanted)
    index = 0
    while index <= last_needed:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if resize_width > 0 and frame.shape[1] != resize_width:
                scale = resize_width / frame.shape[1]
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=interpolation,
                )
            frames[index] = frame.astype(np.float32) / 255.0
        index += 1
    capture.release()

    missing = sorted(wanted.difference(frames))
    if missing:
        raise ValueError(
            f"Video {path} ended before required frame(s) {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    return frames, reported_count


def _stack(frames: dict[int, Any], indices: list[int], np):
    return np.stack([frames[index] for index in indices], axis=0)


def _pixel_metrics(reference, prediction, structural_similarity, np) -> tuple[float, float]:
    if reference.shape != prediction.shape:
        raise ValueError(f"Metric input shape mismatch: {reference.shape} != {prediction.shape}")
    mse = np.mean((reference - prediction) ** 2, axis=(1, 2, 3), dtype=np.float64)
    psnr_per_frame = np.full(mse.shape, 100.0, dtype=np.float64)
    nonzero = mse > 0.0
    psnr_per_frame[nonzero] = 10.0 * np.log10(1.0 / mse[nonzero])
    psnr_per_frame = np.minimum(psnr_per_frame, 100.0)
    ssim_per_frame = [
        structural_similarity(ref, pred, data_range=1.0, channel_axis=-1)
        for ref, pred in zip(reference, prediction)
    ]
    return float(np.mean(psnr_per_frame)), float(np.mean(ssim_per_frame))


def _lpips_tensor(video, torch):
    # LPIPS expects NCHW RGB tensors in [-1, 1].
    return torch.from_numpy(video.transpose(0, 3, 1, 2).copy()).float().mul_(2.0).sub_(1.0)


def _lpips_pairs(left, right, model, device: str, batch_size: int, torch, np):
    if left.shape != right.shape:
        raise ValueError(f"LPIPS pair shape mismatch: {left.shape} != {right.shape}")
    left_tensor = _lpips_tensor(left, torch)
    right_tensor = _lpips_tensor(right, torch)
    values = []
    with torch.inference_mode():
        for start in range(0, len(left_tensor), batch_size):
            end = min(start + batch_size, len(left_tensor))
            score = model(
                left_tensor[start:end].to(device),
                right_tensor[start:end].to(device),
            )
            values.extend(score.reshape(end - start, -1).mean(dim=1).detach().cpu().tolist())
    return np.asarray(values, dtype=np.float64)


def _lpips_matrix(query, reference, model, device: str, batch_size: int, torch, np):
    query_tensor = _lpips_tensor(query, torch)
    reference_tensor = _lpips_tensor(reference, torch)
    num_query = len(query_tensor)
    num_reference = len(reference_tensor)
    total_pairs = num_query * num_reference
    matrix = np.empty((num_query, num_reference), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            flat_indices = np.arange(start, end, dtype=np.int64)
            query_indices = flat_indices // num_reference
            reference_indices = flat_indices % num_reference
            score = model(
                query_tensor[query_indices].to(device),
                reference_tensor[reference_indices].to(device),
            )
            matrix.reshape(-1)[start:end] = (
                score.reshape(end - start, -1).mean(dim=1).detach().cpu().numpy()
            )
    return matrix


def _init_lpips(device: str, net: str):
    import torch
    import lpips

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"LPIPS device {device!r} requested, but CUDA is unavailable")
    model = lpips.LPIPS(net=net, spatial=False).eval().to(device)
    return model, device, torch


def _evaluate_video(
    *,
    video_path: Path,
    closure_pair: list[int],
    resize_width: int,
    max_frames_per_segment: int,
    lpips_state,
    lpips_batch_size: int,
    cv2,
    np,
    structural_similarity,
) -> dict[str, Any]:
    if len(closure_pair) != 2:
        raise ValueError(f"Expected two closure frame indices, got {closure_pair!r}")
    closure_start, closure_end = map(int, closure_pair)
    path_length = closure_end - closure_start
    if path_length < 4 or path_length % 2:
        raise ValueError(
            f"Closure interval must have a positive even length, got {closure_pair!r}"
        )

    turnaround = closure_start + path_length // 2
    reference_full = list(range(closure_start, turnaround))
    revisit_full = list(range(turnaround + 1, closure_end + 1))
    if len(reference_full) != len(revisit_full):
        raise AssertionError("Internal loop segments do not have equal lengths")

    sample_positions = _uniform_indices(len(reference_full), max_frames_per_segment)
    reference_indices = [reference_full[position] for position in sample_positions]
    revisit_indices = [revisit_full[position] for position in sample_positions]
    selected = sorted(set(reference_indices + revisit_indices + [closure_start, closure_end]))
    frames, reported_frame_count = _read_selected_video_frames(
        video_path, selected, resize_width, cv2, np
    )

    reference = _stack(frames, reference_indices, np)
    revisit = _stack(frames, revisit_indices, np)
    closure_reference = _stack(frames, [closure_start], np)
    closure_revisit = _stack(frames, [closure_end], np)

    closure_psnr, closure_ssim = _pixel_metrics(
        closure_reference, closure_revisit, structural_similarity, np
    )
    result: dict[str, Any] = {
        "reported_video_frames": reported_frame_count,
        "closure_start_frame": closure_start,
        "turnaround_frame": turnaround,
        "closure_end_frame": closure_end,
        "full_segment_frames": len(reference_full),
        "metric_segment_frames": len(sample_positions),
        "psnr": None,
        "ssim": None,
        "lpips": None,
        "closure_psnr": closure_psnr,
        "closure_ssim": closure_ssim,
        "closure_lpips": None,
        "mag_psnr": None,
        "mag_ssim": None,
        "mag_lpips": None,
        "match_unique_ratio": None,
        "match_temporal_mae_normalized": None,
        "match_reverse_violation_ratio": None,
        "mag_match_reference_indices": [],
    }

    if lpips_state is None:
        return result

    model, device, torch = lpips_state
    closure_distances = _lpips_pairs(
        closure_revisit,
        closure_reference,
        model,
        device,
        lpips_batch_size,
        torch,
        np,
    )
    distance_matrix = _lpips_matrix(
        revisit, reference, model, device, lpips_batch_size, torch, np
    )
    matched_positions = np.argmin(distance_matrix, axis=1)
    matched_reference = reference[matched_positions]
    mag_psnr, mag_ssim = _pixel_metrics(
        matched_reference, revisit, structural_similarity, np
    )

    expected_positions = np.asarray(
        [len(reference_full) - 1 - position for position in sample_positions],
        dtype=np.int64,
    )
    candidate_positions = np.asarray(sample_positions, dtype=np.int64)
    matched_full_positions = candidate_positions[matched_positions]
    normalizer = max(1, len(reference_full) - 1)
    reverse_violations = (
        float(np.mean(np.diff(matched_full_positions) > 0))
        if len(matched_full_positions) > 1
        else 0.0
    )

    result.update(
        {
            "closure_lpips": float(closure_distances.mean()),
            "psnr": mag_psnr,
            "ssim": mag_ssim,
            "lpips": float(distance_matrix.min(axis=1).mean()),
            "mag_psnr": mag_psnr,
            "mag_ssim": mag_ssim,
            "mag_lpips": float(distance_matrix.min(axis=1).mean()),
            "match_unique_ratio": float(len(np.unique(matched_positions)) / len(matched_positions)),
            "match_temporal_mae_normalized": float(
                np.mean(np.abs(matched_full_positions - expected_positions)) / normalizer
            ),
            "match_reverse_violation_ratio": reverse_violations,
            "mag_match_reference_indices": [
                int(reference_indices[position]) for position in matched_positions.tolist()
            ],
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MAG-style visual loop closure for minWM videos."
    )
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--duration-label",
        default=None,
        help="Manifest duration key such as 10s; inferred from num_output_frames by default.",
    )
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument(
        "--max-frames-per-segment",
        type=int,
        default=96,
        help="Uniformly subsample each path segment to this many frames; 0 keeps every frame.",
    )
    parser.add_argument("--device", default="auto", help="LPIPS device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-batch-size", type=int, default=64)
    parser.add_argument("--skip-lpips", action="store_true")
    parser.add_argument(
        "--require-lpips",
        action="store_true",
        help="Fail instead of retaining only endpoint PSNR/SSIM when LPIPS cannot be initialized.",
    )
    args = parser.parse_args()

    if args.skip_lpips and args.require_lpips:
        parser.error("--skip-lpips and --require-lpips are mutually exclusive")
    if args.lpips_batch_size <= 0:
        parser.error("--lpips-batch-size must be positive")
    if args.max_frames_per_segment < 0:
        parser.error("--max-frames-per-segment cannot be negative")

    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise SystemExit(
            "Loop-closure evaluation requires numpy, opencv-python, and scikit-image. "
            f"Original import error: {exc}"
        ) from exc

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lpips_state = None
    lpips_error = None
    if not args.skip_lpips:
        try:
            lpips_state = _init_lpips(args.device, args.lpips_net)
        except Exception as exc:  # Keep strict pixel metrics available in lightweight environments.
            lpips_error = repr(exc)
            if args.require_lpips:
                raise

    rows = [
        row
        for row in _load_timing(args.timing_csv)
        if row.get("status") in {"generated", "skipped_exists"}
    ]
    metric_rows: list[dict[str, Any]] = []
    errors = []
    for row in rows:
        raw_video_path = row.get("output_path", "")
        video_path = _resolve_video_path(raw_video_path, args.timing_csv)
        if not video_path.is_file():
            errors.append({"output_path": raw_video_path, "error": "missing video"})
            continue
        try:
            prompt_index = int(row.get("prompt_index", ""))
            duration_label = _duration_label(manifest, row, args.duration_label)
            sample = _manifest_sample(manifest, duration_label, prompt_index)
            metrics = _evaluate_video(
                video_path=video_path,
                closure_pair=sample["closure_pair_decoded"],
                resize_width=max(0, int(args.resize_width)),
                max_frames_per_segment=int(args.max_frames_per_segment),
                lpips_state=lpips_state,
                lpips_batch_size=int(args.lpips_batch_size),
                cv2=cv2,
                np=np,
                structural_similarity=structural_similarity,
            )
        except Exception as exc:
            errors.append({"output_path": str(video_path), "error": repr(exc)})
            continue

        metric_rows.append(
            {
                "sample_order": row.get("sample_order", ""),
                "prompt_index": prompt_index,
                "duration_label": duration_label,
                "prompt": row.get("prompt", ""),
                "output_path": str(video_path),
                "last_chunk_fps": _to_float(row.get("last_chunk_fps")),
                "peak_vram_allocated_gb": _to_float(row.get("peak_vram_allocated_gb")),
                **metrics,
            }
        )

    fieldnames = [
        "sample_order",
        "prompt_index",
        "duration_label",
        "prompt",
        "output_path",
        "last_chunk_fps",
        "peak_vram_allocated_gb",
        "reported_video_frames",
        "closure_start_frame",
        "turnaround_frame",
        "closure_end_frame",
        "full_segment_frames",
        "metric_segment_frames",
        *METRIC_FIELDS,
        "mag_match_reference_indices",
    ]
    csv_path = args.output_dir / "loop_closure_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            output_row = dict(row)
            output_row["mag_match_reference_indices"] = json.dumps(
                row["mag_match_reference_indices"], separators=(",", ":")
            )
            writer.writerow(output_row)

    summary_fields = [
        "last_chunk_fps",
        "peak_vram_allocated_gb",
        *METRIC_FIELDS,
    ]
    summary = {
        "timing_csv": str(args.timing_csv),
        "manifest": str(args.manifest),
        "num_evaluated": len(metric_rows),
        "num_errors": len(errors),
        "lpips_available": lpips_state is not None,
        "lpips_device": lpips_state[1] if lpips_state is not None else None,
        "lpips_error": lpips_error,
        "metrics": {
            field: _mean([row.get(field) for row in metric_rows]) for field in summary_fields
        },
        "protocol": {
            "split": (
                "The manifest closure interval is split at its midpoint. The turnaround frame is "
                "excluded; outbound frames before it are the internal reference and revisit frames "
                "after it are the evaluated segment."
            ),
            "mag": (
                "Each revisit frame independently selects its LPIPS-nearest outbound frame; PSNR "
                "and SSIM use those same pairs. The top-level psnr, ssim, and lpips fields are "
                "this MAG-style result; mag_psnr, mag_ssim, and mag_lpips are retained as aliases. "
                "Higher PSNR/SSIM and lower LPIPS are better."
            ),
            "diagnostics": (
                "Higher match_unique_ratio is better; lower match_temporal_mae_normalized and "
                "match_reverse_violation_ratio are better."
            ),
            "subsampling": (
                f"At most {args.max_frames_per_segment} uniformly spaced frames per segment "
                "(0 means all frames)."
            ),
            "limitation": (
                "The outbound segment is a pseudo-reference generated by the same rollout, not an "
                "external ground-truth video. Metrics measure visual memory/closure consistency and "
                "must not be reported as reconstruction fidelity to real data."
            ),
        },
        "errors": errors,
        "videos": metric_rows,
    }
    json_path = args.output_dir / "loop_closure_metrics.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if lpips_state is None:
        print(
            "LPIPS was unavailable; endpoint PSNR/SSIM were retained, but MAG-style PSNR/SSIM/LPIPS "
            "matching was skipped."
        )


if __name__ == "__main__":
    main()
