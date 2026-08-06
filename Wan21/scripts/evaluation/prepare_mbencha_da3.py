#!/usr/bin/env python3
"""Generate and validate MBench-A Depth Anything 3 artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


DA3_SUBSETS = {"environment", "object", "causal"}


@dataclass(frozen=True)
class Task:
    model_id: str
    subset: str
    sample_id: str
    condition_id: str
    video: Path
    output: Path

    @property
    def key(self) -> str:
        return f"{self.model_id}/{self.subset}/{self.sample_id}/{self.condition_id}"


def _rows(path: Path):
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("item_id") or ":".join(
            str(row[name]) for name in ("subset", "sample_id", "condition_id")
        )
        if key in seen:
            continue
        seen.add(key)
        yield lineno, row


def discover_tasks(dataset_root: Path, model_glob: str, subsets: set[str]) -> list[Task]:
    tasks: list[Task] = []
    models_root = dataset_root / "models"
    for model_root in sorted(path for path in models_root.glob(model_glob) if path.is_dir()):
        samples = model_root / "samples.jsonl"
        if not samples.is_file():
            continue
        for lineno, row in _rows(samples):
            subset = str(row.get("subset", ""))
            if subset not in subsets:
                continue
            videos = row.get("media", {}).get("videos", [])
            generated = next((v for v in videos if v.get("role") == "generated"), None)
            if not generated or not generated.get("path"):
                raise ValueError(f"{samples}:{lineno}: generated video is missing")
            video = Path(generated["path"])
            if not video.is_absolute():
                video = model_root / video
            sample_id = str(row["sample_id"])
            condition_id = str(row["condition_id"])
            output = (
                model_root / "artifacts" / subset / sample_id / condition_id / "da3"
            )
            tasks.append(Task(model_root.name, subset, sample_id, condition_id, video, output))
    return tasks


def validate_artifact(root: Path) -> tuple[bool, str]:
    npz_path = root / "results.npz"
    image_dir = root / "input_images"
    if not npz_path.is_file():
        return False, "missing results.npz"
    images = sorted(image_dir.glob("*.png")) if image_dir.is_dir() else []
    if not images:
        return False, "missing input_images/*.png"
    try:
        with np.load(npz_path) as data:
            required = {"depth", "extrinsics", "intrinsics"}
            if not required.issubset(data.files):
                return False, f"missing npz keys {sorted(required - set(data.files))}"
            depth = data["depth"]
            extrinsics = data["extrinsics"]
            intrinsics = data["intrinsics"]
            n = depth.shape[0] if depth.ndim == 3 else -1
            if depth.ndim != 3:
                return False, f"depth shape must be [N,H,W], got {depth.shape}"
            if extrinsics.shape not in {(n, 3, 4), (n, 4, 4)}:
                return False, f"invalid extrinsics shape {extrinsics.shape}"
            if intrinsics.shape != (n, 3, 3):
                return False, f"invalid intrinsics shape {intrinsics.shape}"
            if len(images) != n:
                return False, f"frame mismatch: depth={n}, images={len(images)}"
            if n < 2:
                return False, "fewer than two frames"
            for name, array in (("depth", depth), ("extrinsics", extrinsics), ("intrinsics", intrinsics)):
                if not np.isfinite(array).all():
                    return False, f"{name} contains NaN/Inf"
            if not (depth > 0).any():
                return False, "depth has no positive values"
    except Exception as exc:
        return False, f"unreadable artifact: {exc}"
    return True, "ok"


def read_video(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"video has fewer than two readable frames: {path}")
    return frames


def w2c_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    ext = np.asarray(extrinsics, dtype=np.float64)
    if ext.ndim != 3 or ext.shape[1:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"DA3 extrinsics must be [N,3,4] or [N,4,4], got {ext.shape}")
    homogeneous = np.broadcast_to(np.eye(4), (len(ext), 4, 4)).copy()
    homogeneous[:, : ext.shape[1], : ext.shape[2]] = ext
    return np.linalg.inv(homogeneous)[:, :3, :4].astype(np.float32)


def _prediction_arrays(prediction) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    images = np.asarray(prediction.processed_images)
    depth = np.asarray(prediction.depth, dtype=np.float32)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)
    extrinsics = w2c_to_c2w(np.asarray(prediction.extrinsics))
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"processed_images must be [N,H,W,3], got {images.shape}")
    n = len(images)
    if depth.shape[0] != n or intrinsics.shape != (n, 3, 3) or extrinsics.shape != (n, 3, 4):
        raise ValueError(
            f"DA3 frame/shape mismatch: images={images.shape}, depth={depth.shape}, "
            f"intrinsics={intrinsics.shape}, extrinsics={extrinsics.shape}"
        )
    if images.dtype != np.uint8:
        scale = 255.0 if np.issubdtype(images.dtype, np.floating) and images.max() <= 1.0 else 1.0
        images = np.clip(images * scale, 0, 255).astype(np.uint8)
    return images, depth, extrinsics, intrinsics


def write_artifact(root: Path, prediction, overwrite: bool) -> None:
    import cv2

    images, depth, extrinsics, intrinsics = _prediction_arrays(prediction)
    temporary = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    image_dir = temporary / "input_images"
    image_dir.mkdir(parents=True)
    for index, rgb in enumerate(images):
        if not cv2.imwrite(str(image_dir / f"{index:06d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to write processed frame {index}")
    np.savez_compressed(
        temporary / "results.npz",
        depth=depth,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
    )
    ok, reason = validate_artifact(temporary)
    if not ok:
        raise RuntimeError(f"new DA3 artifact failed validation: {reason}")
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        if not overwrite:
            raise FileExistsError(f"artifact already exists but is invalid: {root}")
        backup = root.with_name(f"{root.name}.invalid-{datetime.now():%Y%m%d-%H%M%S}")
        root.rename(backup)
    temporary.rename(root)


def load_model(model_name: str, device: str):
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError(
            "Depth Anything 3 is not installed. Clone the official repository and run "
            "`pip install -e .` in the selected DA3 environment."
        ) from exc
    model = DepthAnything3.from_pretrained(model_name)
    return model.to(device=device)


def log_event(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_progress(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    subsets = {part.strip() for part in args.subsets.split(",") if part.strip()}
    unsupported = subsets - DA3_SUBSETS
    if unsupported:
        raise ValueError(f"DA3 is not required for MBench-A subsets: {sorted(unsupported)}")
    tasks = discover_tasks(args.dataset_root.resolve(), args.model_glob, subsets)
    tasks = [task for index, task in enumerate(tasks) if index % args.world_size == args.rank]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise RuntimeError("no matching MBench-A DA3 tasks")

    counts = {"valid": 0, "missing": 0, "generated": 0, "failed": 0}
    pending: list[Task] = []
    for task in tasks:
        ok, _ = validate_artifact(task.output)
        if ok:
            counts["valid"] += 1
        else:
            counts["missing"] += 1
            pending.append(task)
    print(f"rank {args.rank}/{args.world_size}: total={len(tasks)} valid={counts['valid']} pending={len(pending)}", flush=True)
    progress = {
        "rank": args.rank,
        "world_size": args.world_size,
        "total": len(tasks),
        "initial_valid": counts["valid"],
        "pending": len(pending),
        "processed": 0,
        "generated": 0,
        "failed": 0,
        "phase": "verify_only" if args.verify_only else "loading_model",
        "current_task": None,
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    write_progress(args.progress_file, progress)
    if args.verify_only:
        return 0 if not pending else 2
    if not pending:
        progress.update(phase="complete", updated_at=time.time())
        write_progress(args.progress_file, progress)
        return 0

    model = load_model(args.model, args.device)
    for position, task in enumerate(pending, 1):
        progress.update(
            phase="running", current_task=task.key, updated_at=time.time()
        )
        write_progress(args.progress_file, progress)
        started = time.monotonic()
        event = {"time": datetime.now().isoformat(), "task": task.key, "rank": args.rank}
        try:
            if not task.video.is_file():
                raise FileNotFoundError(task.video)
            frames = read_video(task.video)
            prediction = model.inference(
                frames,
                process_res=args.process_res,
                process_res_method=args.process_res_method,
                ref_view_strategy=args.ref_view_strategy,
            )
            write_artifact(task.output, prediction, args.overwrite_invalid)
            counts["generated"] += 1
            event.update(status="generated", seconds=round(time.monotonic() - started, 3), frames=len(frames))
        except Exception as exc:
            counts["failed"] += 1
            event.update(status="failed", seconds=round(time.monotonic() - started, 3), error=repr(exc))
        log_event(args.log_file, event)
        progress.update(
            processed=position,
            generated=counts["generated"],
            failed=counts["failed"],
            current_task=None,
            updated_at=time.time(),
        )
        write_progress(args.progress_file, progress)
        print(f"[{position}/{len(pending)}] {task.key}: {event['status']}", flush=True)
    progress.update(phase="complete", current_task=None, updated_at=time.time())
    write_progress(args.progress_file, progress)
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    return 1 if counts["failed"] else 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-root", type=Path, required=True)
    result.add_argument("--model-glob", default="minwm_sink_rebase_retrieval_*_seed0")
    result.add_argument("--subsets", default="environment,object,causal")
    result.add_argument("--model", default="depth-anything/DA3-LARGE-1.1")
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--process-res", type=int, default=504)
    result.add_argument("--process-res-method", default="upper_bound_resize")
    result.add_argument("--ref-view-strategy", default="first")
    result.add_argument("--rank", type=int, default=0)
    result.add_argument("--world-size", type=int, default=1)
    result.add_argument("--limit", type=int)
    result.add_argument("--verify-only", action="store_true")
    result.add_argument("--overwrite-invalid", action="store_true")
    result.add_argument("--log-file", type=Path, default=Path("outputs/mbencha_da3/events.jsonl"))
    result.add_argument("--progress-file", type=Path)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise SystemExit("require world-size >= 1 and 0 <= rank < world-size")
    try:
        raise SystemExit(run(args))
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
