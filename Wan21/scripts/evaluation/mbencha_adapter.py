#!/usr/bin/env python3
"""Prepare minWM inputs from MBench-A and package generated videos for MBench."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


SUPPORTED_ACTIONS = {
    "left_then_right",
    "right_then_left",
    "forward_then_backward",
    "left_360",
    "right_360",
    "left_720",
    "right_720",
    "left_1080",
    "right_1080",
    "static",
}


def _read_dataset_id(dataset_root: Path) -> str:
    config_path = dataset_root / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"MBench dataset config not found: {config_path}")
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("dataset_id:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"dataset_id is missing from {config_path}")


def _conditions(raw: str, length: str) -> list[str]:
    result = []
    for value in raw.split(","):
        condition = value.strip()
        if not condition:
            continue
        if condition.endswith(("_10s", "_25s")):
            condition_length = condition.rsplit("_", 1)[-1]
            if condition_length != length:
                raise ValueError(
                    f"Condition {condition!r} does not match --length {length!r}"
                )
        else:
            condition = f"{condition}_{length}"
        action = condition[: -(len(length) + 1)]
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(
                f"Unsupported MBench-A action {action!r}; "
                f"choose from {sorted(SUPPORTED_ACTIONS)}"
            )
        result.append(condition)
    return result


def _trajectory(action: str, num_frames: int) -> str:
    steps = num_frames - 1
    if steps < 1:
        raise ValueError("--num-output-frames must be at least 2")

    if action == "static":
        return f"n*{steps}"

    roundtrips = {
        "left_then_right": ("j", "l"),
        "right_then_left": ("l", "j"),
        "forward_then_backward": ("w", "s"),
    }
    if action in roundtrips:
        outward, backward = roundtrips[action]
        half = steps // 2
        segments = [f"{outward}*{half}", f"{backward}*{half}"]
        if steps % 2:
            segments.append("n*1")
        return ",".join(segments)

    direction, degrees_raw = action.split("_", 1)
    degrees = int(degrees_raw)
    key = "j" if direction == "left" else "l"
    # A base minWM rotation step is 3 degrees. The multiplier makes the
    # requested total turn exact regardless of rollout length.
    multiplier = degrees / (3.0 * steps)
    return f"{key}@{multiplier:.10g}*{steps}"


def _iter_samples(dataset_root: Path, subsets: set[str]):
    samples_root = dataset_root / "samples"
    if not samples_root.is_dir():
        raise FileNotFoundError(f"MBench samples directory not found: {samples_root}")
    for subset_dir in sorted(path for path in samples_root.iterdir() if path.is_dir()):
        if subsets and subset_dir.name not in subsets:
            continue
        for sample_dir in sorted(path for path in subset_dir.iterdir() if path.is_dir()):
            meta_path = sample_dir / "sample.json"
            if not meta_path.is_file():
                continue
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            caption = meta.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Missing caption in {meta_path}")
            yield subset_dir.name, sample_dir.name, caption.strip()


def _load_official_assignments(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Official MBench-A assignment manifest not found: {path}. "
            "Download MBench-A/models/hy_worldplay/samples.jsonl or use "
            "--cartesian explicitly for a custom condition sweep."
        )
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"subset", "sample_id", "condition_id"} - row.keys()
        if missing:
            raise ValueError(f"{path}:{lineno} is missing fields: {sorted(missing)}")
        rows.append(row)
    return rows


def prepare(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    if _read_dataset_id(dataset_root) != "mbencha":
        raise ValueError(f"{dataset_root} is not an MBench-A dataset")
    conditions = _conditions(args.conditions, args.length)
    condition_filter = set(conditions)
    subsets = {value.strip() for value in args.subsets.split(",") if value.strip()}
    sample_captions = {
        (subset, sample_id): caption
        for subset, sample_id, caption in _iter_samples(dataset_root, subsets)
    }
    rows = []
    if args.cartesian:
        if not conditions:
            raise ValueError("--conditions is required with --cartesian")
        assigned_items = (
            (subset, sample_id, condition_id)
            for subset, sample_id in sample_captions
            for condition_id in conditions
        )
    else:
        assignment_path = (
            args.assignment_manifest.resolve()
            if args.assignment_manifest
            else dataset_root / "models" / "hy_worldplay" / "samples.jsonl"
        )
        assigned_items = (
            (row["subset"], row["sample_id"], row["condition_id"])
            for row in _load_official_assignments(assignment_path)
            if (not subsets or row["subset"] in subsets)
            and (not condition_filter or row["condition_id"] in condition_filter)
        )

    for subset, sample_id, condition_id in assigned_items:
        caption = sample_captions.get((subset, sample_id))
        if caption is None:
            raise ValueError(
                f"Assignment {subset}/{sample_id}/{condition_id} has no matching sample.json"
            )
        condition_length = condition_id.rsplit("_", 1)[-1]
        if condition_length != args.length:
            raise ValueError(
                f"Condition {condition_id!r} does not match --length {args.length!r}"
            )
        action = condition_id[: -(len(args.length) + 1)]
        if action not in SUPPORTED_ACTIONS:
            raise ValueError(f"Unsupported MBench-A action in assignment: {action!r}")
        rows.append(
            {
                "prompt_index": len(rows),
                "subset": subset,
                "sample_id": sample_id,
                "condition_id": condition_id,
                "prompt": caption,
                "trajectory": _trajectory(action, args.num_output_frames),
            }
        )
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No MBench-A samples matched the requested filters")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    (args.work_dir / "prompts.txt").write_text(
        "".join(f"{row['prompt']}\n" for row in rows), encoding="utf-8"
    )
    (args.work_dir / "trajectories.txt").write_text(
        "".join(f"{row['trajectory']}\n" for row in rows), encoding="utf-8"
    )
    with (args.work_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Prepared {len(rows)} generation items in {args.work_dir}")


def _load_manifest(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[int(row["prompt_index"])] = row
    return result


def _link_video(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        destination.symlink_to(os.path.relpath(source, destination.parent))


def package(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    if _read_dataset_id(dataset_root) != "mbencha":
        raise ValueError(f"{dataset_root} is not an MBench-A dataset")
    manifest = _load_manifest(args.manifest)
    run_root = args.run_root.resolve()
    timing_files = sorted(run_root.glob("seed_*/*/inference_times.csv"))
    if not timing_files:
        raise FileNotFoundError(f"No case outputs found below {run_root}")

    packages: dict[str, list[dict]] = {}
    for timing_path in timing_files:
        seed_name = timing_path.parents[1].name
        case_name = timing_path.parent.name
        seed = seed_name.removeprefix("seed_")
        model_id = f"{args.model_prefix}_{case_name}_seed{seed}"
        with timing_path.open(newline="", encoding="utf-8") as handle:
            # A resumed rollout may append the same prompt more than once.  Keep
            # the last successful row so packaging is idempotent even before a
            # timing CSV is normalized in place.
            successful: dict[int, dict[str, str]] = {}
            for timing in csv.DictReader(handle):
                if timing.get("status") not in {"generated", "skipped_exists"}:
                    continue
                prompt_index = int(timing["prompt_index"])
                successful[prompt_index] = timing
        for prompt_index, timing in sorted(successful.items()):
            if prompt_index not in manifest:
                raise ValueError(
                    f"prompt_index={prompt_index} from {timing_path} "
                    "is absent from the adapter manifest"
                )
            source = Path(timing["output_path"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Generated video not found: {source}")
            item = manifest[prompt_index]
            subset = item["subset"]
            sample_id = item["sample_id"]
            condition_id = item["condition_id"]
            relative_video = (
                Path("outputs") / subset / sample_id / condition_id / "video.mp4"
            )
            model_root = dataset_root / "models" / model_id
            _link_video(source, model_root / relative_video, args.link_mode)
            row = {
                    "item_id": f"{subset}:{sample_id}:{condition_id}",
                    "dataset_id": "mbencha",
                    "subset": subset,
                    "sample_id": sample_id,
                    "condition_id": condition_id,
                    "model_id": model_id,
                    "media": {
                        "videos": [{"path": str(relative_video), "role": "generated"}]
                    },
                    "artifacts": {},
                    "annotations": {},
                    "metadata": {
                        "minwm_case": case_name,
                        "seed": int(seed),
                        "source_video": str(source),
                    },
            }
            packages.setdefault(model_id, []).append(row)

    for model_id, rows in packages.items():
        samples_path = dataset_root / "models" / model_id / "samples.jsonl"
        samples_path.parent.mkdir(parents=True, exist_ok=True)
        with samples_path.open("w", encoding="utf-8") as handle:
            for row in sorted(
                rows, key=lambda value: (value["subset"], value["sample_id"], value["condition_id"])
            ):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Packaged {len(rows)} items as model {model_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prep = commands.add_parser("prepare", help="Create minWM prompt/trajectory files")
    prep.add_argument("--dataset-root", type=Path, required=True)
    prep.add_argument("--work-dir", type=Path, required=True)
    prep.add_argument("--length", choices=["10s", "25s"], required=True)
    prep.add_argument(
        "--conditions",
        default="",
        help="Optional condition filter; empty means all official assignments.",
    )
    prep.add_argument(
        "--assignment-manifest",
        type=Path,
        help="Official samples.jsonl; defaults to models/hy_worldplay/samples.jsonl.",
    )
    prep.add_argument(
        "--cartesian",
        action="store_true",
        help="Custom ablation mode: combine every selected sample with every condition.",
    )
    prep.add_argument("--subsets", default="environment,human,object,causal")
    prep.add_argument("--num-output-frames", type=int, required=True)
    prep.add_argument("--limit", type=int)
    prep.set_defaults(func=prepare)

    pack = commands.add_parser("package", help="Create MBench model packages")
    pack.add_argument("--dataset-root", type=Path, required=True)
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--run-root", type=Path, required=True)
    pack.add_argument("--model-prefix", default="minwm")
    pack.add_argument("--link-mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    pack.set_defaults(func=package)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
