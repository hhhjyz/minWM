#!/usr/bin/env python3
"""Summarize official VBench outputs into CSV and LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
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


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _case_label(case_dir: Path, roots: list[Path]) -> tuple[str, str, str]:
    for root in roots:
        try:
            rel = case_dir.relative_to(root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) == 1:
            return root.name, "seed_0", parts[0]
        if len(parts) >= 2 and parts[0].startswith("seed_"):
            return root.name, parts[0], parts[1]
        return root.name, "-", "/".join(parts)
    return case_dir.parent.name, "-", case_dir.name


def collect_rows(roots: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        for metrics_csv in sorted(root.rglob("eval/official_vbench_metrics.csv")):
            case_dir = metrics_csv.parent.parent
            if case_dir in seen:
                continue
            seen.add(case_dir)
            with metrics_csv.open(newline="", encoding="utf-8") as f:
                video_rows = list(csv.DictReader(f))
            group, seed, case = _case_label(case_dir, roots)
            row: dict[str, Any] = {
                "group": group,
                "seed": seed,
                "case": case,
                "num_videos": len(video_rows),
                "case_dir": str(case_dir),
            }
            for dim in DIMENSIONS:
                values = [_float_or_none(v.get(dim)) for v in video_rows]
                row[dim] = _mean([v for v in values if v is not None])
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "seed", "case", "num_videos", *DIMENSIONS, "case_dir"]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "--"
    if abs(number) >= 10:
        return f"{number:.2f}"
    return f"{number:.4f}"


def _latex_escape(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def _table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{_latex_escape(title)}}}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Seed & Case & N & Subject & Background & Flicker & Smooth & Dynamic & Aesthetic & Imaging \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["seed"]),
                    _latex_escape(row["case"]),
                    str(row["num_videos"]),
                    _fmt(row["subject_consistency"]),
                    _fmt(row["background_consistency"]),
                    _fmt(row["temporal_flickering"]),
                    _fmt(row["motion_smoothness"]),
                    _fmt(row["dynamic_degree"]),
                    _fmt(row["aesthetic_quality"]),
                    _fmt(row["imaging_quality"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _aggregate_across_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["group"], row["case"]), []).append(row)

    output = []
    for (group, case), group_rows in sorted(grouped.items()):
        if len(group_rows) < 2:
            continue
        row: dict[str, Any] = {
            "group": group,
            "seed": "mean",
            "case": case,
            "num_videos": sum(int(r["num_videos"]) for r in group_rows),
        }
        for dim in DIMENSIONS:
            values = [_float_or_none(r.get(dim)) for r in group_rows]
            row[dim] = _mean([v for v in values if v is not None])
        output.append(row)
    return output


def write_latex(rows: list[dict[str, Any]], output_tex: Path, roots: list[Path]) -> None:
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    doc = [
        r"\documentclass{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{hyperref}",
        r"\title{minWM Official VBench Evaluation Summary}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Notes}",
        "The tables summarize official VBench raw quality dimensions for generated videos. "
        "They are not final VBench leaderboard scores because semantic prompt dimensions were not run. "
        "All seven configured quality dimensions, including \\texttt{dynamic\\_degree}, are reported.",
        "",
        "Evaluated roots:",
        r"\begin{itemize}",
    ]
    for root in roots:
        doc.append(rf"\item \texttt{{{_latex_escape(root)}}}")
    doc.extend([r"\end{itemize}", ""])

    for group in sorted(grouped):
        group_rows = sorted(grouped[group], key=lambda r: (r["seed"], r["case"]))
        doc.append(_table(group, group_rows))

    aggregate = _aggregate_across_seeds(rows)
    if aggregate:
        doc.append(_table("Mean Across Available Seeds", aggregate))

    doc.extend([r"\end{document}", ""])
    output_tex.write_text("\n".join(doc), encoding="utf-8")


def compile_pdf(tex_path: Path) -> Path | None:
    engines = ["latexmk", "pdflatex", "xelatex", "lualatex", "tectonic"]
    engine = next((item for item in engines if shutil.which(item)), None)
    if engine is None:
        return None
    if engine == "latexmk":
        cmd = [engine, "-pdf", "-interaction=nonstopmode", tex_path.name]
    elif engine == "tectonic":
        cmd = [engine, tex_path.name]
    else:
        cmd = [engine, "-interaction=nonstopmode", tex_path.name]
    subprocess.run(cmd, cwd=str(tex_path.parent), check=False)
    pdf_path = tex_path.with_suffix(".pdf")
    return pdf_path if pdf_path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--compile-pdf", action="store_true")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.root]
    rows = collect_rows(roots)
    if not rows:
        raise RuntimeError("No official_vbench_metrics.csv files found.")

    output_csv = args.output_dir / "official_vbench_summary.csv"
    output_json = args.output_dir / "official_vbench_summary.json"
    output_tex = args.output_dir / "official_vbench_report.tex"
    write_csv(rows, output_csv)
    output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_latex(rows, output_tex, roots)
    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    print(f"Wrote {output_tex}")

    if args.compile_pdf:
        pdf_path = compile_pdf(output_tex)
        if pdf_path:
            print(f"Wrote {pdf_path}")
        else:
            print("No LaTeX engine found or PDF was not produced; kept .tex output.")


if __name__ == "__main__":
    main()
