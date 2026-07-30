#!/usr/bin/env python3
"""Build a validated Markdown report for minWM string-loop experiments."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any


LOOP_METRICS = [
    "psnr",
    "ssim",
    "lpips",
    "closure_psnr",
    "closure_ssim",
    "closure_lpips",
    "match_unique_ratio",
    "match_temporal_mae_normalized",
    "match_reverse_violation_ratio",
]
VBENCH_METRICS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value: {value!r}")
    return number


def index_rows(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate {key} values")
    return indexed


def fmt(value: float, metric: str) -> str:
    if metric in {"psnr", "closure_psnr", "imaging_quality"}:
        return f"{value:.3f}"
    return f"{value:.4f}"


def signed(value: float, metric: str) -> str:
    return f"{value:+.3f}" if metric in {"psnr", "closure_psnr", "imaging_quality"} else f"{value:+.4f}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_case(case_dir: Path) -> dict[str, Any]:
    timing_rows = read_csv(case_dir / "inference_times.csv")
    loop_rows = read_csv(case_dir / "eval" / "loop_closure_metrics.csv")
    vbench_rows = read_csv(case_dir / "eval" / "official_vbench_metrics.csv")
    if len(timing_rows) != 30 or len(loop_rows) != 30 or len(vbench_rows) != 30:
        raise ValueError(
            f"{case_dir.name}: expected 30 timing/loop/VBench rows, got "
            f"{len(timing_rows)}/{len(loop_rows)}/{len(vbench_rows)}"
        )

    timing_by_name = index_rows(timing_rows, "output_path")
    timing_names = {Path(path).name for path in timing_by_name}
    loop_names = {Path(row["output_path"]).name for row in loop_rows}
    vbench_names = {Path(row["video_path"]).name for row in vbench_rows}
    if timing_names != loop_names or timing_names != vbench_names:
        raise ValueError(f"{case_dir.name}: timing/loop/VBench video sets do not match")

    for row in loop_rows:
        for metric in LOOP_METRICS:
            finite(row[metric])
    for row in vbench_rows:
        for metric in VBENCH_METRICS:
            finite(row[metric])

    loop_by_name = {Path(row["output_path"]).name: row for row in loop_rows}
    vbench_by_name = {Path(row["video_path"]).name: row for row in vbench_rows}
    timing_by_name = {Path(path).name: row for path, row in timing_by_name.items()}
    return {
        "name": case_dir.name,
        "timing": timing_by_name,
        "loop": loop_by_name,
        "vbench": vbench_by_name,
    }


def metric_mean(case: dict[str, Any], group: str, metric: str) -> float:
    return mean(finite(row[metric]) for row in case[group].values())


def paired_delta(case: dict[str, Any], baseline: dict[str, Any], group: str, metric: str) -> float:
    names = sorted(baseline[group])
    if set(names) != set(case[group]):
        raise ValueError(f"{case['name']}: video set differs from baseline for {group}")
    return mean(
        finite(case[group][name][metric]) - finite(baseline[group][name][metric])
        for name in names
    )


def best_case(cases: list[dict[str, Any]], group: str, metric: str, higher: bool = True) -> tuple[str, float]:
    values = [(case["name"], metric_mean(case, group, metric)) for case in cases]
    return (max if higher else min)(values, key=lambda item: item[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    case_dirs = sorted(
        path for path in (root / "seed_0").iterdir() if (path / "inference_times.csv").is_file()
    )
    cases = [load_case(path) for path in case_dirs]
    by_name = {case["name"]: case for case in cases}
    if "baseline" not in by_name:
        raise ValueError("baseline case is missing")
    baseline = by_name["baseline"]

    manifest = {}
    manifest_path = root / "experiment_manifest.txt"
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                manifest[key] = value
    duration = f"{int(manifest.get('num_output_frames', '0')) // 4}s"

    loop_rows = []
    for case in cases:
        loop_rows.append(
            [
                case["name"],
                "30",
                fmt(metric_mean(case, "loop", "psnr"), "psnr"),
                signed(paired_delta(case, baseline, "loop", "psnr"), "psnr"),
                fmt(metric_mean(case, "loop", "ssim"), "ssim"),
                signed(paired_delta(case, baseline, "loop", "ssim"), "ssim"),
                fmt(metric_mean(case, "loop", "lpips"), "lpips"),
                signed(paired_delta(case, baseline, "loop", "lpips"), "lpips"),
                fmt(metric_mean(case, "loop", "match_unique_ratio"), "match_unique_ratio"),
                fmt(metric_mean(case, "loop", "match_temporal_mae_normalized"), "match_temporal_mae_normalized"),
                fmt(metric_mean(case, "loop", "match_reverse_violation_ratio"), "match_reverse_violation_ratio"),
            ]
        )

    closure_rows = []
    for case in cases:
        closure_rows.append(
            [
                case["name"],
                fmt(metric_mean(case, "loop", "closure_psnr"), "closure_psnr"),
                signed(paired_delta(case, baseline, "loop", "closure_psnr"), "closure_psnr"),
                fmt(metric_mean(case, "loop", "closure_ssim"), "closure_ssim"),
                signed(paired_delta(case, baseline, "loop", "closure_ssim"), "closure_ssim"),
                fmt(metric_mean(case, "loop", "closure_lpips"), "closure_lpips"),
                signed(paired_delta(case, baseline, "loop", "closure_lpips"), "closure_lpips"),
            ]
        )

    vbench_rows = []
    vbench_delta_rows = []
    for case in cases:
        vbench_rows.append(
            [case["name"], "30"]
            + [fmt(metric_mean(case, "vbench", metric), metric) for metric in VBENCH_METRICS]
        )
        vbench_delta_rows.append(
            [case["name"]]
            + [signed(paired_delta(case, baseline, "vbench", metric), metric) for metric in VBENCH_METRICS]
        )

    timing_rows = []
    for case in cases:
        timing = case["timing"].values()
        total_seconds = mean(finite(row["total_seconds"]) for row in timing)
        fps = mean(finite(row["last_chunk_fps"]) for row in timing)
        peak = max(finite(row["peak_vram_allocated_gb"]) for row in timing)
        timing_rows.append(
            [
                case["name"],
                f"{total_seconds:.2f}",
                f"{total_seconds - mean(finite(r['total_seconds']) for r in baseline['timing'].values()):+.2f}",
                f"{fps:.4f}",
                f"{peak:.2f}",
            ]
        )

    winners = [
        ("MAG PSNR ↑", *best_case(cases, "loop", "psnr")),
        ("MAG SSIM ↑", *best_case(cases, "loop", "ssim")),
        ("MAG LPIPS ↓", *best_case(cases, "loop", "lpips", False)),
        ("Subject consistency ↑", *best_case(cases, "vbench", "subject_consistency")),
        ("Background consistency ↑", *best_case(cases, "vbench", "background_consistency")),
        ("Temporal flickering ↑", *best_case(cases, "vbench", "temporal_flickering")),
        ("Motion smoothness ↑", *best_case(cases, "vbench", "motion_smoothness")),
        ("Aesthetic quality ↑", *best_case(cases, "vbench", "aesthetic_quality")),
        ("Imaging quality ↑", *best_case(cases, "vbench", "imaging_quality")),
    ]

    report = [
        f"# minWM String-loop {duration} Evaluation Report",
        "",
        f"- Root: `{root}`",
        f"- Cases: {len(cases)}; videos per case: 30; seed: 0",
        "- Loop metrics: `minwm-fa`, LPIPS-Alex, 256 px evaluation width, MAG-style nearest outbound matching.",
        "- VBench metrics: official VBench `custom_input`, local checkpoints.",
        "- `Δ` is the paired per-video mean difference relative to `baseline`.",
        "- Direction: PSNR/SSIM/VBench/unique ratio ↑; LPIPS/time/temporal MAE/reverse violations ↓. Dynamic degree describes motion amount and is not a pure quality score.",
        "",
        "## Metric leaders",
        "",
        md_table(
            ["Metric", "Best case", "Value"],
            [[label, name, fmt(value, "imaging_quality" if "Imaging" in label else "ssim")] for label, name, value in winners],
        ),
        "",
        "## Loop closure: MAG-style path consistency",
        "",
        md_table(
            ["Case", "N", "PSNR ↑", "Δ PSNR", "SSIM ↑", "Δ SSIM", "LPIPS ↓", "Δ LPIPS", "Unique ↑", "Temporal MAE ↓", "Reverse ↓"],
            loop_rows,
        ),
        "",
        "## Loop closure: exact endpoint",
        "",
        md_table(
            ["Case", "Closure PSNR ↑", "Δ", "Closure SSIM ↑", "Δ", "Closure LPIPS ↓", "Δ"],
            closure_rows,
        ),
        "",
        "## Official VBench",
        "",
        md_table(
            ["Case", "N", "Subject ↑", "Background ↑", "Flicker ↑", "Smooth ↑", "Dynamic", "Aesthetic ↑", "Imaging ↑"],
            vbench_rows,
        ),
        "",
        "## Official VBench paired delta vs baseline",
        "",
        md_table(
            ["Case", "Δ Subject", "Δ Background", "Δ Flicker", "Δ Smooth", "Δ Dynamic", "Δ Aesthetic", "Δ Imaging"],
            vbench_delta_rows,
        ),
        "",
        "## Inference efficiency",
        "",
        md_table(["Case", "Total s/video ↓", "Δ seconds", "Last-chunk FPS ↑", "Peak VRAM GB ↓"], timing_rows),
        "",
        "## Interpretation notes",
        "",
        "- Loop PSNR/SSIM/LPIPS use the outbound part of the same generated rollout as an internal pseudo-reference; they measure visual closure/memory consistency, not fidelity to real-video ground truth.",
        "- Official VBench values are averaged over the same 30 prompts for every case. Imaging quality uses the official VBench scale, which differs from the 0–1 consistency dimensions.",
        "- Only seed 0 is available, so small differences should be treated as preliminary until repeated across seeds.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
