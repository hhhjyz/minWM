"""Pose/FOV retrieval helpers for WorldKV-style KV windows."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def normalize_retrieval_metric(metric: str) -> str:
    metric = str(metric or "pose").lower().strip()
    aliases = {
        "fov": "worldkv_fov",
        "worldkv": "worldkv_fov",
        "hy": "hy_fov",
        "hyworldplay": "hy_fov",
    }
    metric = aliases.get(metric, metric)
    valid = {"recent_only", "pose", "worldkv_fov", "hy_fov", "hybrid"}
    if metric not in valid:
        raise ValueError(f"Unknown retrieval_metric={metric!r}; expected one of {sorted(valid)}")
    return metric


def generate_deterministic_points_in_sphere(
    n_points: int,
    radius: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Deterministic approximately uniform probe points inside a sphere."""
    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    idx = torch.arange(n_points, device=device, dtype=dtype) + 0.5
    frac = idx / float(n_points)
    z = 1.0 - 2.0 * frac
    theta = idx * (math.pi * (3.0 - math.sqrt(5.0)))
    xy = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    shell = torch.stack((xy * torch.cos(theta), xy * torch.sin(theta), z), dim=-1)
    radii = radius * torch.pow(frac, 1.0 / 3.0)
    return shell * radii[:, None]


def _normalize_distances(distances: torch.Tensor) -> torch.Tensor:
    max_val = distances.max()
    if max_val > 0:
        return distances / max_val
    return distances


def _as_c2w(viewmats_w2c: torch.Tensor) -> torch.Tensor:
    if viewmats_w2c.ndim == 4:
        viewmats_w2c = viewmats_w2c[0]
    return torch.linalg.inv(viewmats_w2c.float())


def _mean_pose_c2w(viewmats_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = _as_c2w(viewmats_w2c)
    return c2w[:, :3, 3].mean(dim=0), c2w[:, :3, :3].mean(dim=0)


def pose_block_distances(
    current_viewmats: torch.Tensor,
    candidate_viewmats: Iterable[torch.Tensor],
) -> torch.Tensor:
    """WorldKV-style normalized translation + rotation chunk distance."""
    current_trans, current_rot = _mean_pose_c2w(current_viewmats)
    trans_parts = []
    rot_parts = []
    for hist_viewmats in candidate_viewmats:
        hist_trans, hist_rot = _mean_pose_c2w(hist_viewmats)
        trans_parts.append(((hist_trans - current_trans) ** 2).sum())
        rel_rot = hist_rot.transpose(-1, -2) @ current_rot
        trace_val = rel_rot.diagonal(dim1=-2, dim2=-1).sum()
        rot_parts.append(torch.acos(((trace_val - 1.0) * 0.5).clamp(-1.0, 1.0)))
    if not trans_parts:
        return torch.empty(0, device=current_viewmats.device)
    trans_dist = torch.stack(trans_parts).to(current_viewmats.device)
    rot_dist = torch.stack(rot_parts).to(current_viewmats.device)
    return 0.5 * _normalize_distances(trans_dist) + 0.5 * _normalize_distances(rot_dist)


def _inside_fov_from_c2w(
    points_world: torch.Tensor,
    c2w: torch.Tensor,
    tan_half_fov_h: float,
    tan_half_fov_v: float,
    max_radius: float | None,
) -> torch.Tensor:
    center = c2w[:3, 3]
    rot_c2w = c2w[:3, :3]
    vectors_world = points_world - center[None, :]
    points_cam = vectors_world @ rot_c2w
    z = points_cam[:, 2]
    safe_z = torch.clamp(z.abs(), min=1e-6)
    mask = (
        (z > 1e-6)
        & ((points_cam[:, 0].abs() / safe_z) < tan_half_fov_h)
        & ((points_cam[:, 1].abs() / safe_z) < tan_half_fov_v)
    )
    if max_radius is not None:
        mask = mask & (torch.linalg.norm(vectors_world, dim=-1) < max_radius)
    return mask


def worldkv_fov_overlap_similarity(
    current_w2c: torch.Tensor,
    hist_w2c: torch.Tensor,
    points_camera: torch.Tensor,
    *,
    fov_h_deg: float,
    fov_v_deg: float,
    radius: float,
) -> torch.Tensor:
    """WorldKV deterministic C2W frustum overlap."""
    current_c2w = torch.linalg.inv(current_w2c.float())
    hist_c2w = torch.linalg.inv(hist_w2c.float())
    fov_h_rad = math.radians(float(fov_h_deg))
    fov_v_rad = math.radians(float(fov_v_deg))
    tan_half_h = math.tan(fov_h_rad / 2.0)
    tan_half_v = math.tan(fov_v_rad / 2.0)
    z = points_camera[:, 2]
    safe_z = torch.clamp(z.abs(), min=1e-6)
    current_mask = (
        (z > 1e-6)
        & ((points_camera[:, 0].abs() / safe_z) < tan_half_h)
        & ((points_camera[:, 1].abs() / safe_z) < tan_half_v)
    )
    current_count = current_mask.sum()
    if current_count == 0:
        return torch.zeros((), device=points_camera.device, dtype=torch.float32)
    points_world = current_c2w[:3, 3][None, :] + points_camera @ current_c2w[:3, :3].T
    hist_mask = _inside_fov_from_c2w(points_world, hist_c2w, tan_half_h, tan_half_v, radius)
    return (current_mask & hist_mask).sum().float() / current_count.float()


def _rotation_matrix_to_pitch_yaw_from_w2c(rotation_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c2w_rot = rotation_w2c.T
    forward = c2w_rot[:, 2]
    yaw = torch.atan2(forward[0], forward[2]) * (180.0 / math.pi)
    pitch = torch.atan2(forward[1], torch.sqrt(forward[0] ** 2 + forward[2] ** 2)) * (180.0 / math.pi)
    return pitch, yaw


def _inside_hy_fov(
    points: torch.Tensor,
    center: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
    half_h_deg: torch.Tensor,
    half_v_deg: torch.Tensor,
) -> torch.Tensor:
    vectors = points - center[None, :]
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    azimuth = torch.atan2(x, z) * (180.0 / math.pi)
    elevation = torch.atan2(y, torch.sqrt(x ** 2 + z ** 2)) * (180.0 / math.pi)
    diff_azimuth = torch.remainder(azimuth - yaw + 180.0, 360.0) - 180.0
    diff_elevation = torch.remainder(elevation - pitch + 180.0, 360.0) - 180.0
    return (diff_azimuth.abs() < half_h_deg) & (diff_elevation.abs() < half_v_deg)


def hy_fov_overlap_similarity(
    current_w2c: torch.Tensor,
    hist_w2c: torch.Tensor,
    points_local: torch.Tensor,
    *,
    fov_h_deg: float,
    fov_v_deg: float,
    radius: float,
) -> torch.Tensor:
    """HY-WorldPlay-style W2C angular FOV overlap."""
    current_w2c = current_w2c.float()
    hist_w2c = hist_w2c.float()
    relative_current_w2c = torch.linalg.inv(current_w2c @ torch.linalg.inv(current_w2c))
    relative_hist_w2c = torch.linalg.inv(current_w2c @ torch.linalg.inv(hist_w2c))

    current_rot, current_t = relative_current_w2c[:3, :3], relative_current_w2c[:3, 3]
    hist_rot, hist_t = relative_hist_w2c[:3, :3], relative_hist_w2c[:3, 3]
    current_center = -current_rot.T @ current_t
    hist_center = -hist_rot.T @ hist_t
    current_pitch, current_yaw = _rotation_matrix_to_pitch_yaw_from_w2c(current_rot)
    hist_pitch, hist_yaw = _rotation_matrix_to_pitch_yaw_from_w2c(hist_rot)
    half_h = torch.tensor(float(fov_h_deg) / 2.0, device=points_local.device)
    half_v = torch.tensor(float(fov_v_deg) / 2.0, device=points_local.device)
    points_world = points_local + current_center[None, :]
    current_mask = _inside_hy_fov(points_world, current_center, current_pitch, current_yaw, half_h, half_v)
    hist_mask = _inside_hy_fov(points_world, hist_center, hist_pitch, hist_yaw, half_h, half_v)
    hist_mask = hist_mask & (torch.linalg.norm(points_world - hist_center[None, :], dim=1) < radius)
    current_count = current_mask.sum()
    if current_count == 0:
        return torch.zeros((), device=points_local.device, dtype=torch.float32)
    return (current_mask & hist_mask).sum().float() / current_count.float()


def fov_block_distances(
    current_viewmats: torch.Tensor,
    candidate_viewmats: Iterable[torch.Tensor],
    *,
    style: str,
    points: torch.Tensor,
    fov_h_deg: float,
    fov_v_deg: float,
    radius: float,
) -> torch.Tensor:
    if current_viewmats.ndim == 4:
        current_viewmats = current_viewmats[0]
    current_mid = min(current_viewmats.shape[0] // 2, current_viewmats.shape[0] - 1)
    distances = []
    for hist_viewmats in candidate_viewmats:
        if hist_viewmats.ndim == 4:
            hist_viewmats = hist_viewmats[0]
        hist_mid = min(hist_viewmats.shape[0] // 2, hist_viewmats.shape[0] - 1)
        if style == "hy_fov":
            sim = hy_fov_overlap_similarity(
                current_viewmats[current_mid],
                hist_viewmats[hist_mid],
                points,
                fov_h_deg=fov_h_deg,
                fov_v_deg=fov_v_deg,
                radius=radius,
            )
        else:
            sim = worldkv_fov_overlap_similarity(
                current_viewmats[current_mid],
                hist_viewmats[hist_mid],
                points,
                fov_h_deg=fov_h_deg,
                fov_v_deg=fov_v_deg,
                radius=radius,
            )
        distances.append(1.0 - sim)
    if not distances:
        return torch.empty(0, device=current_viewmats.device)
    return torch.stack(distances).to(current_viewmats.device)


def retrieval_block_distances(
    current_viewmats: torch.Tensor,
    candidate_viewmats: list[torch.Tensor],
    *,
    metric: str,
    points: torch.Tensor | None = None,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 35.0,
    fov_radius: float = 8.0,
    hybrid_fov_weight: float = 0.5,
) -> torch.Tensor:
    metric = normalize_retrieval_metric(metric)
    if metric == "recent_only":
        return torch.empty(0, device=current_viewmats.device)
    if metric == "pose":
        return pose_block_distances(current_viewmats, candidate_viewmats)
    if points is None:
        raise ValueError(f"{metric} retrieval requires FOV probe points")
    if metric in {"worldkv_fov", "hy_fov"}:
        return fov_block_distances(
            current_viewmats,
            candidate_viewmats,
            style=metric,
            points=points,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            radius=fov_radius,
        )

    pose_dist = pose_block_distances(current_viewmats, candidate_viewmats)
    fov_dist = fov_block_distances(
        current_viewmats,
        candidate_viewmats,
        style="worldkv_fov",
        points=points,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        radius=fov_radius,
    )
    fov_weight = min(1.0, max(0.0, float(hybrid_fov_weight)))
    return (1.0 - fov_weight) * _normalize_distances(pose_dist) + fov_weight * fov_dist


def select_topk_indices(candidate_indices: list[int], distances: torch.Tensor, k: int) -> list[int]:
    if k <= 0 or not candidate_indices:
        return []
    values = distances.detach().float().cpu().tolist()
    ranked = sorted(zip(candidate_indices, values), key=lambda item: (item[1], item[0]))
    return [index for index, _ in ranked[: min(k, len(ranked))]]
