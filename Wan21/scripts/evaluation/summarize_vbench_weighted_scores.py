#!/usr/bin/env python3
"""Compute official VBench quality scores and write Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

# VBench scripts/constant.py (T2V leaderboard quality-score protocol).
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
}
WEIGHTS = {
    "subject_consistency": 1.0,
    "background_consistency": 1.0,
    "temporal_flickering": 1.0,
    "motion_smoothness": 1.0,
    "dynamic_degree": 0.5,
    "aesthetic_quality": 1.0,
    "imaging_quality": 1.0,
}


def quality_score(summary: dict[str, Any]) -> tuple[float, dict[str, float]]:
    normalized: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for dim in DIMENSIONS:
        value = float(summary[dim])
        minimum, maximum = NORMALIZATION[dim]
        score = (value - minimum) / (maximum - minimum)
        normalized[dim] = score
        weighted_sum += score * WEIGHTS[dim]
        total_weight += WEIGHTS[dim]
    return weighted_sum / total_weight, normalized


def experiment_root(outputs_root: Path, case_dir: Path) -> Path:
    rel = case_dir.relative_to(outputs_root)
    parts = rel.parts
    if len(parts) >= 3 and parts[1].startswith("seed_"):
        return outputs_root / parts[0]
    if len(parts) >= 2:
        return outputs_root / parts[0]
    return case_dir


def case_label(root: Path, case_dir: Path) -> str:
    if root == case_dir:
        return case_dir.name
    return str(case_dir.relative_to(root))


def collect(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_json in sorted(outputs_root.rglob("eval/official_vbench_metrics.json")):
        case_dir = metrics_json.parent.parent
        metrics_csv = metrics_json.with_suffix(".csv")
        if not metrics_csv.is_file():
            continue
        with metrics_csv.open(newline="", encoding="utf-8") as f:
            video_count = sum(1 for _ in csv.DictReader(f))
        payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        summary = payload.get("summary", {})
        if video_count <= 0 or any(dim not in summary for dim in DIMENSIONS):
            continue
        score, normalized = quality_score(summary)
        root = experiment_root(outputs_root, case_dir)
        rows.append(
            {
                "experiment": str(root.relative_to(outputs_root)),
                "case": case_label(root, case_dir),
                "num_videos": video_count,
                "quality_score": score,
                "quality_score_percent": 100.0 * score,
                **{dim: float(summary[dim]) for dim in DIMENSIONS},
                **{f"normalized_{dim}": normalized[dim] for dim in DIMENSIONS},
                "case_dir": str(case_dir),
                "experiment_root": str(root),
            }
        )
    return rows


def markdown(rows: list[dict[str, Any]], *, title: str, global_table: bool) -> str:
    ordered = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
    lines = [
        f"# {title}",
        "",
        "该表仅包含已经完成全部七个官方质量维度的 case。`加权质量总分` 严格采用",
        "VBench `scripts/cal_final_score.py` 的 Quality Score 公式：先按官方 min/max",
        "归一化各维度，再以 `Subject=1, Background=1, Flickering=1, Smoothness=1,",
        "Dynamic=0.5, Aesthetic=1, Imaging=1` 加权平均。",
        "",
        "> 当前实验未评测九个 semantic dimensions，因此不能计算官方 leaderboard",
        "> Total Score；下表总分是官方 Quality Score（百分制），不是完整 Total Score。",
        "",
    ]
    if global_table:
        lines.extend(
            [
                "| 排名 | 实验 | Case | N | 加权质量总分 ↑ |",
                "|---:|---|---|---:|---:|",
            ]
        )
        for rank, row in enumerate(ordered, 1):
            lines.append(
                f"| {rank} | `{row['experiment']}` | `{row['case']}` | "
                f"{row['num_videos']} | **{row['quality_score_percent']:.4f}** |"
            )
    else:
        lines.extend(
            [
                "| 排名 | Case | N | Subject | Background | Flickering | Smoothness | Dynamic | Aesthetic | Imaging | 加权质量总分 ↑ |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, row in enumerate(ordered, 1):
            lines.append(
                f"| {rank} | `{row['case']}` | {row['num_videos']} | "
                f"{row['subject_consistency']:.4f} | {row['background_consistency']:.4f} | "
                f"{row['temporal_flickering']:.4f} | {row['motion_smoothness']:.4f} | "
                f"{row['dynamic_degree']:.4f} | {row['aesthetic_quality']:.4f} | "
                f"{row['imaging_quality']:.4f} | **{row['quality_score_percent']:.4f}** |"
            )
    lines.append("")
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "experiment",
        "case",
        "num_videos",
        *DIMENSIONS,
        "quality_score",
        "quality_score_percent",
        *[f"normalized_{dim}" for dim in DIMENSIONS],
        "case_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["quality_score"], reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    outputs_root = args.outputs_root.resolve()
    rows = collect(outputs_root)
    if not rows:
        raise RuntimeError("No complete official VBench quality results found")

    (outputs_root / "official_vbench_weighted_summary.md").write_text(
        markdown(rows, title="全部官方 VBench 加权质量总分", global_table=True),
        encoding="utf-8",
    )
    write_csv(rows, outputs_root / "official_vbench_weighted_summary.csv")

    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[Path(row["experiment_root"])].append(row)
    for root, group_rows in grouped.items():
        root.mkdir(parents=True, exist_ok=True)
        (root / "official_vbench_weighted_summary.md").write_text(
            markdown(
                group_rows,
                title=f"{root.name}：官方 VBench 加权质量总分",
                global_table=False,
            ),
            encoding="utf-8",
        )
    print(f"Wrote {len(rows)} cases across {len(grouped)} experiment summaries")


if __name__ == "__main__":
    main()
