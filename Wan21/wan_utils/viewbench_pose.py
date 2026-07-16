"""Utilities for ViewBench camera-pose adaptation.

ViewBench stores per-frame camera poses from UE5.  minWM's camera path expects
world-to-camera matrices and intrinsics tensors, so the small benchmark builder
converts each selected sequence into a compact ``.npz`` with:

  - viewmats: (T, 4, 4) world-to-camera matrices
  - Ks:       (T, 3, 3) normalized intrinsics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_K = np.array(
    [[0.5050505, 0.0, 0.5], [0.0, 0.89786756, 0.5], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


# UE camera convention from ViewBench README:
#   X=forward, Y=right, Z=up
# minWM/WorldPlay camera tensors are treated as OpenCV-like:
#   X=right, Y=down, Z=forward
# Columns map OpenCV camera basis vectors into the UE camera basis.
UE_FROM_OPENCV_CAMERA = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


def _as_matrix4(value: Any) -> np.ndarray | None:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    return None


def _rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _c2w_from_position_rotation(record: dict[str, Any]) -> np.ndarray | None:
    position = (
        record.get("position")
        or record.get("location")
        or record.get("translation")
        or record.get("pos")
    )
    rotation = (
        record.get("rotation")
        or record.get("rot")
        or record.get("euler")
        or record.get("rotation_euler")
    )
    if position is None or rotation is None:
        return None

    if isinstance(position, dict):
        t = np.array(
            [
                position.get("x", position.get("X", 0.0)),
                position.get("y", position.get("Y", 0.0)),
                position.get("z", position.get("Z", 0.0)),
            ],
            dtype=np.float32,
        )
    else:
        t = np.asarray(position, dtype=np.float32).reshape(-1)[:3]

    if isinstance(rotation, dict):
        pitch = rotation.get("pitch", rotation.get("Pitch", 0.0))
        roll = rotation.get("roll", rotation.get("Roll", 0.0))
        yaw = rotation.get("yaw", rotation.get("Yaw", 0.0))
    else:
        # ViewBench README: rotation = [pitch, roll, yaw] in degrees.
        pitch, roll, yaw = np.asarray(rotation, dtype=np.float32).reshape(-1)[:3]

    pitch, roll, yaw = np.radians([pitch, roll, yaw])
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)
    c2w[:3, 3] = t
    return c2w


def _extract_frames(data: Any) -> list[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("frames", "poses", "camera_poses", "cameras", "trajectory"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        # Some JSON dumps use frame ids as top-level keys.
        numeric_items = []
        for key, value in data.items():
            try:
                numeric_items.append((int(key), value))
            except (TypeError, ValueError):
                pass
        if numeric_items:
            return [value for _, value in sorted(numeric_items)]
    raise ValueError("Cannot find a frame pose list in ViewBench JSON")


def _extract_c2w(record: Any) -> np.ndarray:
    if isinstance(record, (list, tuple)):
        mat = _as_matrix4(record)
        if mat is not None:
            return mat
    if not isinstance(record, dict):
        raise ValueError(f"Unsupported pose record type: {type(record)!r}")

    for key in (
        "c2w",
        "camera_to_world",
        "camera_to_world_matrix",
        "transform_matrix",
        "matrix",
        "pose",
    ):
        if key in record:
            mat = _as_matrix4(record[key])
            if mat is not None:
                return mat

    nested = record.get("camera") or record.get("Camera")
    if isinstance(nested, dict):
        try:
            return _extract_c2w(nested)
        except ValueError:
            pass

    mat = _c2w_from_position_rotation(record)
    if mat is not None:
        return mat
    raise ValueError(f"Cannot extract c2w from pose record keys={list(record.keys())}")


def load_viewbench_json_c2w(json_path: str | Path) -> np.ndarray:
    """Load ViewBench per-frame camera-to-world matrices as ``(T, 4, 4)``."""

    with Path(json_path).open(encoding="utf-8") as f:
        data = json.load(f)
    frames = _extract_frames(data)
    c2w = np.stack([_extract_c2w(frame) for frame in frames]).astype(np.float32)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape (T,4,4), got {c2w.shape}")
    return c2w


def c2w_ue_to_viewmats(
    c2w_ue: np.ndarray,
    *,
    translation_scale: float = 0.01,
    convert_camera_axes: bool = True,
    recenter: bool = True,
) -> np.ndarray:
    """Convert ViewBench UE c2w matrices to minWM-style w2c view matrices.

    ``translation_scale=0.01`` converts ViewBench centimeters to meters.  The
    optional recentering keeps the first pose near identity, which is closer to
    minWM's built-in trajectory generator and avoids huge absolute translations.
    """

    c2w = np.asarray(c2w_ue, dtype=np.float32).copy()
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected c2w shape (T,4,4), got {c2w.shape}")

    c2w[:, :3, 3] *= float(translation_scale)
    if convert_camera_axes:
        axis = np.eye(4, dtype=np.float32)
        axis[:3, :3] = UE_FROM_OPENCV_CAMERA
        c2w = c2w @ axis

    if recenter:
        first_inv = np.linalg.inv(c2w[0])
        c2w = first_inv[None] @ c2w

    return np.linalg.inv(c2w).astype(np.float32)


def make_intrinsics(num_frames: int, k: np.ndarray | None = None) -> np.ndarray:
    base = DEFAULT_K if k is None else np.asarray(k, dtype=np.float32)
    if base.shape != (3, 3):
        raise ValueError(f"Expected intrinsics shape (3,3), got {base.shape}")
    return np.repeat(base[None], int(num_frames), axis=0).astype(np.float32)


def save_pose_npz(path: str | Path, viewmats: np.ndarray, Ks: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    viewmats = np.asarray(viewmats, dtype=np.float32)
    if Ks is None:
        Ks = make_intrinsics(len(viewmats))
    np.savez_compressed(path, viewmats=viewmats, Ks=np.asarray(Ks, dtype=np.float32))


def load_pose_npz(
    path: str | Path,
    *,
    num_frames: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Load a ViewBench/minWM pose npz as batched camera tensors.

    If ``num_frames`` is positive and shorter than the stored pose sequence, the
    pose is truncated.  If it is longer, this raises because padding would break
    loop-closure metrics.
    """

    data = np.load(Path(path))
    viewmats = np.asarray(data["viewmats"], dtype=np.float32)
    Ks = np.asarray(data["Ks"], dtype=np.float32) if "Ks" in data.files else make_intrinsics(len(viewmats))
    if viewmats.ndim != 3 or viewmats.shape[1:] != (4, 4):
        raise ValueError(f"{path}: expected viewmats shape (T,4,4), got {viewmats.shape}")
    if Ks.shape != (len(viewmats), 3, 3):
        raise ValueError(f"{path}: expected Ks shape ({len(viewmats)},3,3), got {Ks.shape}")

    stored_frames = len(viewmats)
    if num_frames is not None and num_frames > 0:
        if stored_frames < num_frames:
            raise ValueError(
                f"{path}: pose has {stored_frames} frames, but num_output_frames={num_frames}"
            )
        viewmats = viewmats[:num_frames]
        Ks = Ks[:num_frames]
    else:
        num_frames = stored_frames

    viewmats_t = torch.from_numpy(viewmats).unsqueeze(0).to(device=device, dtype=dtype)
    Ks_t = torch.from_numpy(Ks).unsqueeze(0).to(device=device, dtype=dtype)
    return viewmats_t, Ks_t, int(num_frames)
