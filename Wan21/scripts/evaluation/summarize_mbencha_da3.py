#!/usr/bin/env python3
"""Summarize MBench-A DA3 artifact completeness as Markdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from prepare_mbencha_da3 import DA3_SUBSETS, discover_tasks, validate_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-glob", default="minwm_sink_rebase_retrieval_*_seed0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tasks = discover_tasks(args.dataset_root.resolve(), args.model_glob, DA3_SUBSETS)
    table: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    reasons: dict[str, int] = defaultdict(int)
    for task in tasks:
        ok, reason = validate_artifact(task.output)
        table[task.model_id][task.subset][0] += int(ok)
        table[task.model_id][task.subset][1] += 1
        if not ok:
            reasons[reason] += 1

    lines = [
        "# MBench-A DA3 artifacts 进度",
        "",
        "| 模型 | environment | object | causal | 总计 |",
        "|---|---:|---:|---:|---:|",
    ]
    valid_total = expected_total = 0
    for model_id in sorted(table):
        cells = []
        model_valid = model_expected = 0
        for subset in ("environment", "object", "causal"):
            valid, expected = table[model_id][subset]
            cells.append(f"{valid}/{expected}")
            model_valid += valid
            model_expected += expected
        valid_total += model_valid
        expected_total += model_expected
        lines.append(f"| {model_id} | {' | '.join(cells)} | {model_valid}/{model_expected} |")
    lines += ["", f"总进度：**{valid_total}/{expected_total}**。"]
    if reasons:
        lines += ["", "缺失/无效原因：", ""]
        lines += [f"- `{reason}`：{count}" for reason, count in sorted(reasons.items())]
    content = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Wrote {args.output}")
    print(content, end="")


if __name__ == "__main__":
    main()
