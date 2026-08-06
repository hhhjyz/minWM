#!/usr/bin/env python3
"""Render aggregate progress for concurrently running MBench DA3 workers."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path


def format_duration(seconds: float) -> str:
    if seconds < 0 or seconds == float("inf"):
        return "--"
    minutes = int(seconds // 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def load_states(paths: list[Path]) -> list[dict]:
    states: list[dict] = []
    for path in paths:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        states.append(state)
    return states


def render(paths: list[Path]) -> bool:
    states = load_states(paths)
    now = time.time()
    if not states:
        print(f"[DA3 {datetime.now():%H:%M:%S}] 等待 {len(paths)} 个 worker 初始化...", flush=True)
        return False
    if len(states) < len(paths):
        print(
            f"[DA3 {datetime.now():%H:%M:%S}] worker 初始化中："
            f"{len(states)}/{len(paths)}...",
            flush=True,
        )
        return False

    total = sum(int(s.get("total", 0)) for s in states)
    initial_valid = sum(int(s.get("initial_valid", 0)) for s in states)
    processed = sum(int(s.get("processed", 0)) for s in states)
    generated = sum(int(s.get("generated", 0)) for s in states)
    failed = sum(int(s.get("failed", 0)) for s in states)
    accounted = initial_valid + processed
    remaining = max(0, total - accounted)
    started = min(float(s.get("started_at", now)) for s in states)
    elapsed = max(1.0, now - started)
    rate_per_min = processed / elapsed * 60.0
    eta = remaining / (rate_per_min / 60.0) if rate_per_min > 0 else float("inf")
    active = sum(s.get("phase") in {"loading_model", "running"} for s in states)
    done = len(states) == len(paths) and all(s.get("phase") == "complete" for s in states)
    percent = accounted / total * 100.0 if total else 0.0
    print(
        f"[DA3 {datetime.now():%H:%M:%S}] {accounted}/{total} ({percent:.2f}%) "
        f"| 新生成={generated} 跳过有效={initial_valid} 失败={failed} "
        f"| active={active}/{len(paths)} | {rate_per_min:.2f} samples/min "
        f"| elapsed={format_duration(elapsed)} ETA={format_duration(eta)}",
        flush=True,
    )
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("progress_files", nargs="+", type=Path)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        done = render(args.progress_files)
        if args.once or done:
            return
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
