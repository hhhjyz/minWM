#!/usr/bin/env python3
"""Run official VBench for multiple minWM experiment roots and summarize."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


DEFAULT_DIMS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def case_dirs(roots: list[Path]):
    seen: set[Path] = set()
    for root in roots:
        for timing_csv in sorted(root.rglob("inference_times.csv")):
            if timing_csv.parent.name == "profile":
                continue
            case_dir = timing_csv.parent.resolve()
            if case_dir in seen:
                continue
            seen.add(case_dir)
            yield case_dir


def write_status(status_csv: Path, rows: list[dict[str, str]]) -> None:
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    with status_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_dir", "status", "log_path"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--minwm-root", type=Path, default=Path.cwd())
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--vbench-python", required=True)
    parser.add_argument("--driver-python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/official_vbench_report"))
    parser.add_argument("--status-csv", type=Path, default=Path("outputs/official_vbench_batch_status.csv"))
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/official_vbench_logs"))
    parser.add_argument("--dimensions", nargs="*", default=DEFAULT_DIMS)
    parser.add_argument("--base-port", type=int, default=42000)
    parser.add_argument("--load-ckpt-from-local", action="store_true")
    parser.add_argument("--vbench-cache-dir", type=Path, default=None)
    parser.add_argument("--compile-pdf", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run cases even when official metrics already exist.")
    args = parser.parse_args()

    minwm_root = args.minwm_root.resolve()
    roots = [root.resolve() for root in args.root]
    log_dir = (minwm_root / args.log_dir).resolve() if not args.log_dir.is_absolute() else args.log_dir
    status_csv = (minwm_root / args.status_csv).resolve() if not args.status_csv.is_absolute() else args.status_csv
    output_dir = (minwm_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for idx, case_dir in enumerate(case_dirs(roots), 1):
        try:
            rel = case_dir.relative_to(minwm_root / "outputs")
        except ValueError:
            rel = Path(case_dir.name)
        metrics_csv = case_dir / "eval" / "official_vbench_metrics.csv"
        metrics_json = case_dir / "eval" / "official_vbench_metrics.json"
        log_path = log_dir / (str(rel).replace("/", "__") + ".log")

        if not args.force and metrics_csv.exists() and metrics_json.exists():
            status = "skipped_existing"
            rows.append({"case_dir": str(case_dir), "status": status, "log_path": str(log_path)})
            write_status(status_csv, rows)
            print(f"[{idx}] {status}: {rel}", flush=True)
            continue

        cmd = [
            args.driver_python,
            "Wan21/scripts/evaluation/evaluate_vbench_official.py",
            "--timing-csv",
            str(case_dir / "inference_times.csv"),
            "--output-dir",
            str(case_dir / "eval"),
            "--vbench-root",
            str(args.vbench_root.resolve()),
            "--python-bin",
            args.vbench_python,
            "--master-port",
            str(args.base_port + idx * 100),
            "--dimensions",
            *args.dimensions,
        ]
        if args.load_ckpt_from_local:
            cmd.append("--load-ckpt-from-local")
        if args.vbench_cache_dir is not None:
            cmd.extend(["--vbench-cache-dir", str(args.vbench_cache_dir.resolve())])
        print(f"[{idx}] official VBench: {rel}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(" ".join(cmd) + "\n")
            log.flush()
            try:
                subprocess.run(cmd, cwd=str(minwm_root), stdout=log, stderr=subprocess.STDOUT, check=True)
                status = "ok"
            except subprocess.CalledProcessError as exc:
                status = f"failed:{exc.returncode}"
        rows.append({"case_dir": str(case_dir), "status": status, "log_path": str(log_path)})
        write_status(status_csv, rows)
        print(f"[{idx}] {status}: {rel}", flush=True)

    summary_cmd = [
        args.driver_python,
        "Wan21/scripts/evaluation/summarize_official_vbench_report.py",
        "--output-dir",
        str(output_dir),
        *sum((["--root", str(root)] for root in roots), []),
    ]
    if args.compile_pdf:
        summary_cmd.append("--compile-pdf")
    subprocess.run(summary_cmd, cwd=str(minwm_root), check=True)


if __name__ == "__main__":
    main()
