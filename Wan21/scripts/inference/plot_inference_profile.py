#!/usr/bin/env python3
"""Summarize per-video inference timing."""

import argparse
import csv
import json
import shutil
from pathlib import Path


NUMERIC_FIELDS = [
    "sample_order",
    "prompt_index",
    "num_output_frames",
    "num_generated_latent_frames",
    "generation_seconds",
    "postprocess_seconds",
    "write_video_seconds",
    "total_seconds",
    "chunk0_latency_seconds",
]


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in NUMERIC_FIELDS:
                if key in parsed:
                    parsed[key] = to_float(parsed[key])
            rows.append(parsed)
    return rows


def load_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def maybe_import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:
        return None, repr(exc)


def generated_rows(rows):
    return [row for row in rows if row.get("status") == "generated"]


def make_summary(rows, run_meta):
    gen = generated_rows(rows)
    total_seconds = [row.get("total_seconds") for row in gen]
    generation_seconds = [row.get("generation_seconds") for row in gen]
    postprocess_seconds = [row.get("postprocess_seconds") for row in gen]
    write_video_seconds = [row.get("write_video_seconds") for row in gen]
    chunk0 = [row.get("chunk0_latency_seconds") for row in gen]
    frames = [row.get("num_output_frames") for row in gen if row.get("num_output_frames") is not None]
    total_frames = sum(frames) if frames else None
    wall = run_meta.get("wall_time_seconds")

    return {
        "run": run_meta,
        "video_count": len(rows),
        "generated_video_count": len(gen),
        "skipped_video_count": len(rows) - len(gen),
        "total_video_seconds_sum": sum(v for v in total_seconds if v is not None),
        "avg_total_seconds_per_video": mean(total_seconds),
        "avg_generation_seconds_per_video": mean(generation_seconds),
        "avg_postprocess_seconds_per_video": mean(postprocess_seconds),
        "avg_write_video_seconds_per_video": mean(write_video_seconds),
        "avg_chunk0_latency_seconds": mean(chunk0),
        "total_requested_frames": total_frames,
        "avg_frames_per_second_by_video_time": (
            total_frames / sum(v for v in total_seconds if v is not None)
            if total_frames and total_seconds
            else None
        ),
        "avg_frames_per_second_by_wall_time": (
            total_frames / wall if total_frames and wall else None
        ),
        "videos": rows,
    }


def plot(rows, out_dir: Path, plt):
    gen = generated_rows(rows)
    if not gen:
        return []

    labels = [str(int(row["sample_order"])) if row.get("sample_order") is not None else str(i) for i, row in enumerate(gen)]
    outputs = []

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, [row.get("total_seconds") or 0.0 for row in gen])
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Total seconds")
    ax.set_title("Per-video inference time")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "video_total_seconds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(str(path))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    generation = [row.get("generation_seconds") or 0.0 for row in gen]
    postprocess = [row.get("postprocess_seconds") or 0.0 for row in gen]
    writing = [row.get("write_video_seconds") or 0.0 for row in gen]
    ax.bar(labels, generation, label="generation")
    ax.bar(labels, postprocess, bottom=generation, label="postprocess")
    bottoms = [a + b for a, b in zip(generation, postprocess)]
    ax.bar(labels, writing, bottom=bottoms, label="write video")
    ax.set_xlabel("Sample order")
    ax.set_ylabel("Seconds")
    ax.set_title("Per-video timing breakdown")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path = out_dir / "video_stage_seconds.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--times-csv", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.times_csv)
    run_meta = load_json(args.run_meta)

    if args.times_csv.exists():
        shutil.copy2(args.times_csv, args.output_dir / "inference_times.csv")

    summary = make_summary(rows, run_meta)
    summary["artifacts"] = {
        "log_file": str(args.log_file),
        "times_csv": str(args.output_dir / "inference_times.csv"),
    }

    plt, plot_error = maybe_import_plotting()
    if plt is not None:
        plots = plot(rows, args.output_dir, plt)
    else:
        plots = []
        summary["plot_error"] = plot_error
    summary["artifacts"]["plots"] = plots

    summary_json = args.output_dir / "profile_summary.json"
    summary_txt = args.output_dir / "profile_summary.txt"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "Inference timing summary",
        f"status: {run_meta.get('exit_status')}",
        f"wall_time_seconds: {run_meta.get('wall_time_seconds')}",
        f"generated_video_count: {summary['generated_video_count']}",
        f"avg_total_seconds_per_video: {summary['avg_total_seconds_per_video']}",
        f"avg_generation_seconds_per_video: {summary['avg_generation_seconds_per_video']}",
        f"avg_postprocess_seconds_per_video: {summary['avg_postprocess_seconds_per_video']}",
        f"avg_write_video_seconds_per_video: {summary['avg_write_video_seconds_per_video']}",
        f"avg_chunk0_latency_seconds: {summary['avg_chunk0_latency_seconds']}",
        f"avg_frames_per_second_by_video_time: {summary['avg_frames_per_second_by_video_time']}",
        f"avg_frames_per_second_by_wall_time: {summary['avg_frames_per_second_by_wall_time']}",
        f"plots: {len(plots)}",
    ]
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_txt}")
    for item in plots:
        print(f"Wrote {item}")


if __name__ == "__main__":
    main()
