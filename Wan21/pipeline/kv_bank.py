"""Chunk-level KV bank for long-horizon minWM inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .debug_utils import is_main_process
from .retrieval_utils import (
    generate_deterministic_points_in_sphere,
    normalize_retrieval_metric,
    retrieval_block_distances,
    select_topk_indices,
)


@dataclass
class KVBankLayerEntry:
    k: torch.Tensor
    v: torch.Tensor
    prope_k: Optional[torch.Tensor]
    prope_v: Optional[torch.Tensor]


@dataclass
class KVBankBlock:
    block_id: int
    frame_start: int
    frame_end: int
    token_count: int
    branches: dict[str, list[KVBankLayerEntry]]
    viewmats: Optional[torch.Tensor]
    Ks: Optional[torch.Tensor]
    pose_summary: Optional[dict[str, float]]
    storage_bytes: int
    stored_compressed: bool = False
    compression_keep_ratio: Optional[float] = None
    motion_score: Optional[float] = None


def _tensor_bytes(tensor: Optional[torch.Tensor]) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _copy_tensor(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.detach().to(device=device).clone()


def normalize_prope_reencode_mode(mode: str) -> str:
    mode = str(mode or "none").lower().strip()
    if mode not in {"none", "current"}:
        raise ValueError(f"Unknown prope_reencode_mode={mode!r}; expected none/current")
    return mode


def _lift_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(Ks.shape[:-2] + (4, 4), device=Ks.device, dtype=Ks.dtype)
    out[..., :3, :3] = Ks
    out[..., 3, 3] = 1.0
    return out


def _invert_K(Ks: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Ks)
    out[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    out[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    out[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    out[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    out[..., 2, 2] = 1.0
    return out


def _invert_SE3(viewmats: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(viewmats)
    Rinv = viewmats[..., :3, :3].transpose(-1, -2)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, viewmats[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def _camera_projection_pair(viewmats: torch.Tensor, Ks: Optional[torch.Tensor]):
    viewmats = viewmats.float()
    if Ks is not None:
        Ks = Ks.float().to(device=viewmats.device)
        Ks_norm = torch.zeros_like(Ks)
        Ks_norm[..., 0, 0] = Ks[..., 0, 0]
        Ks_norm[..., 1, 1] = Ks[..., 1, 1]
        Ks_norm[..., 2, 2] = 1.0
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_inv = torch.einsum("...ij,...jk->...ik", _invert_SE3(viewmats), _lift_K(_invert_K(Ks_norm)))
    else:
        P = viewmats
        P_inv = _invert_SE3(viewmats)
    return P, P_inv


def _apply_camera_matrix_to_prope(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Apply a per-camera 4x4 PRoPE matrix to [B, tokens, heads, dim]."""
    batch, tokens, heads, dim = feats.shape
    cameras = matrix.shape[1]
    if tokens < cameras or tokens % cameras != 0 or dim % 4 != 0:
        return feats
    feats_bhld = feats.permute(0, 2, 1, 3)
    out = torch.einsum(
        "bcij,bhcpkj->bhcpki",
        matrix.to(device=feats.device, dtype=feats.dtype),
        feats_bhld.reshape(batch, heads, cameras, -1, dim // 4, 4),
    ).reshape(feats_bhld.shape)
    return out.permute(0, 2, 1, 3)


def reencode_prope_kv_to_current(
    prope_k: torch.Tensor,
    prope_v: torch.Tensor,
    *,
    src_viewmats: Optional[torch.Tensor],
    src_Ks: Optional[torch.Tensor],
    dst_viewmats: Optional[torch.Tensor],
    dst_Ks: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Re-encode PRoPE KV from source camera poses to current camera poses.

    Stored PRoPE KV is approximately ``P_src_inv @ raw_kv``. Re-encoding uses
    ``P_dst_inv @ P_src @ stored_kv``. The operation is only applied when the
    source and destination camera counts match and token layout is frame-regular.
    """
    if src_viewmats is None or dst_viewmats is None:
        return prope_k, prope_v, False
    src_viewmats = src_viewmats.detach().float()
    dst_viewmats = dst_viewmats.detach().float()
    if src_viewmats.ndim == 3:
        src_viewmats = src_viewmats.unsqueeze(0)
    if dst_viewmats.ndim == 3:
        dst_viewmats = dst_viewmats.unsqueeze(0)
    if src_viewmats.shape[:2] != dst_viewmats.shape[:2]:
        return prope_k, prope_v, False
    if prope_k.shape[1] % src_viewmats.shape[1] != 0:
        return prope_k, prope_v, False

    src_Ks = src_Ks.detach().float() if src_Ks is not None else None
    dst_Ks = dst_Ks.detach().float() if dst_Ks is not None else None
    if src_Ks is not None and src_Ks.ndim == 3:
        src_Ks = src_Ks.unsqueeze(0)
    if dst_Ks is not None and dst_Ks.ndim == 3:
        dst_Ks = dst_Ks.unsqueeze(0)
    if (src_Ks is None) != (dst_Ks is None):
        return prope_k, prope_v, False
    if src_Ks is not None and src_Ks.shape[:2] != src_viewmats.shape[:2]:
        return prope_k, prope_v, False
    if dst_Ks is not None and dst_Ks.shape[:2] != dst_viewmats.shape[:2]:
        return prope_k, prope_v, False

    src_P, _ = _camera_projection_pair(src_viewmats, src_Ks)
    _, dst_P_inv = _camera_projection_pair(dst_viewmats, dst_Ks)
    transform = torch.einsum("...ij,...jk->...ik", dst_P_inv, src_P)
    return (
        _apply_camera_matrix_to_prope(prope_k, transform),
        _apply_camera_matrix_to_prope(prope_v, transform),
        True,
    )


def _latest_cache_slice(cache: dict, token_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    local_end = int(cache["local_end_index"].item())
    local_start = local_end - token_count
    if local_start < 0:
        raise RuntimeError(
            f"KV bank cannot extract {token_count} tokens from cache with local_end={local_end}"
        )
    return cache["k"][:, local_start:local_end], cache["v"][:, local_start:local_end]


def compress_retrieval_kv(
    retr_k: torch.Tensor,
    retr_v: torch.Tensor,
    *,
    chunk_size: int,
    frame_seq_length: int,
    keep_ratio: float,
    anchor_rotate: bool = False,
    pooled: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """WorldKV-style anchor + novelty token pruning for retrieved KV."""

    batch, total_tokens, heads, dim = retr_k.shape
    chunk_tokens = int(chunk_size) * int(frame_seq_length)
    if total_tokens == 0 or chunk_tokens <= 0 or total_tokens % chunk_tokens != 0:
        return retr_k, retr_v

    keep_per_frame = max(1, int(math.ceil(float(keep_ratio) * frame_seq_length)))
    if keep_per_frame >= frame_seq_length:
        return retr_k, retr_v

    num_chunks = total_tokens // chunk_tokens
    out_k_chunks = []
    out_v_chunks = []
    eps = 1e-8

    for chunk_index in range(num_chunks):
        start = chunk_index * chunk_tokens
        chunk_k = retr_k[:, start:start + chunk_tokens].view(
            batch, chunk_size, frame_seq_length, heads, dim
        )
        chunk_v = retr_v[:, start:start + chunk_tokens].view(
            batch, chunk_size, frame_seq_length, heads, dim
        )

        anchor_offset = (chunk_index % chunk_size) if anchor_rotate else 0
        anchor_k = chunk_k[:, anchor_offset]
        anchor_v = chunk_v[:, anchor_offset]
        centroid = anchor_k.float().mean(dim=1)
        centroid_sq = (centroid ** 2).sum(dim=(-2, -1))

        if pooled:
            non_anchor_k = torch.cat(
                [chunk_k[:, frame] for frame in range(chunk_size) if frame != anchor_offset],
                dim=1,
            )
            non_anchor_v = torch.cat(
                [chunk_v[:, frame] for frame in range(chunk_size) if frame != anchor_offset],
                dim=1,
            )
            keep_total = (chunk_size - 1) * keep_per_frame
            non_anchor_float = non_anchor_k.float()
            dot = (non_anchor_float * centroid.unsqueeze(1)).sum(dim=(-2, -1))
            token_sq = (non_anchor_float ** 2).sum(dim=(-2, -1))
            sim = dot / (torch.sqrt(token_sq) * torch.sqrt(centroid_sq).unsqueeze(-1) + eps)
            _, indices = sim.topk(keep_total, dim=-1, largest=False)
            indices, _ = indices.sort(dim=-1)
            gather_idx = indices[:, :, None, None].expand(-1, -1, heads, dim)
            out_k_chunks.append(torch.cat([anchor_k, torch.gather(non_anchor_k, 1, gather_idx)], dim=1))
            out_v_chunks.append(torch.cat([anchor_v, torch.gather(non_anchor_v, 1, gather_idx)], dim=1))
            continue

        frames_k = [None] * chunk_size
        frames_v = [None] * chunk_size
        frames_k[anchor_offset] = anchor_k
        frames_v[anchor_offset] = anchor_v
        for frame in range(chunk_size):
            if frame == anchor_offset:
                continue
            frame_k = chunk_k[:, frame]
            frame_v = chunk_v[:, frame]
            frame_float = frame_k.float()
            dot = (frame_float * centroid.unsqueeze(1)).sum(dim=(-2, -1))
            token_sq = (frame_float ** 2).sum(dim=(-2, -1))
            sim = dot / (torch.sqrt(token_sq) * torch.sqrt(centroid_sq).unsqueeze(-1) + eps)
            _, indices = sim.topk(keep_per_frame, dim=-1, largest=False)
            indices, _ = indices.sort(dim=-1)
            gather_idx = indices[:, :, None, None].expand(-1, -1, heads, dim)
            frames_k[frame] = torch.gather(frame_k, 1, gather_idx)
            frames_v[frame] = torch.gather(frame_v, 1, gather_idx)

        out_k_chunks.append(torch.cat(frames_k, dim=1))
        out_v_chunks.append(torch.cat(frames_v, dim=1))

    return torch.cat(out_k_chunks, dim=1), torch.cat(out_v_chunks, dim=1)


def _camera_metadata(viewmats, Ks):
    stored_viewmats = None
    stored_Ks = None
    summary = None
    if viewmats is not None:
        stored_viewmats = viewmats.detach().float().cpu().clone()
        c2w = torch.linalg.inv(stored_viewmats)
        start = c2w[:, 0]
        end = c2w[:, -1]
        translation = torch.linalg.norm(end[:, :3, 3] - start[:, :3, 3], dim=-1).mean()
        relative_rotation = start[:, :3, :3].transpose(-1, -2) @ end[:, :3, :3]
        trace = relative_rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
        rotation = torch.acos(cosine).mean()
        summary = {
            "translation_delta": float(translation.item()),
            "rotation_delta_rad": float(rotation.item()),
        }
    if Ks is not None:
        stored_Ks = Ks.detach().float().cpu().clone()
    return stored_viewmats, stored_Ks, summary


class KVBank:
    """Store clean block KV without changing the active attention window."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        device: str = "cpu",
        max_blocks: int = 0,
        log_interval: int = 1,
        warn_memory_gb: float = 16.0,
        compression_enable: bool = False,
        compression_keep_ratio: float = 0.5,
        compression_at_store: bool = False,
        compression_pooled: bool = False,
        compression_dynamic_enable: bool = False,
        compression_dynamic_min_keep: float = 0.25,
        compression_dynamic_max_keep: float = 0.75,
        compression_dynamic_translation_scale: float = 1.0,
        compression_dynamic_rotation_scale: float = 0.35,
        compression_dynamic_motion_weight: float = 0.25,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported KV bank device: {device!r}")
        self.enabled = bool(enabled)
        self.device_name = device
        self.max_blocks = max(0, int(max_blocks))
        self.log_interval = max(1, int(log_interval))
        self.warn_memory_bytes = max(0, int(float(warn_memory_gb) * 1024 ** 3))
        self.compression_enable = bool(compression_enable)
        self.compression_keep_ratio = float(compression_keep_ratio)
        self.compression_at_store = bool(compression_at_store)
        self.compression_pooled = bool(compression_pooled)
        self.compression_dynamic_enable = bool(compression_dynamic_enable)
        self.compression_dynamic_min_keep = float(compression_dynamic_min_keep)
        self.compression_dynamic_max_keep = float(compression_dynamic_max_keep)
        self.compression_dynamic_translation_scale = max(1e-6, float(compression_dynamic_translation_scale))
        self.compression_dynamic_rotation_scale = max(1e-6, float(compression_dynamic_rotation_scale))
        self.compression_dynamic_motion_weight = float(compression_dynamic_motion_weight)
        self.blocks: list[KVBankBlock] = []
        self.total_bytes = 0
        self.evicted_blocks = 0
        self.expected_blocks = 0
        self._printed_projection = False

    def reset(self, *, expected_blocks: int = 0) -> None:
        self.blocks.clear()
        self.total_bytes = 0
        self.evicted_blocks = 0
        self.expected_blocks = max(0, int(expected_blocks))
        self._printed_projection = False

    def __len__(self) -> int:
        return len(self.blocks)

    def _storage_device(self, source_device: torch.device) -> torch.device:
        return source_device if self.device_name == "cuda" else torch.device("cpu")

    def _effective_keep_ratio(self, pose_summary: Optional[dict[str, float]]) -> tuple[float, Optional[float]]:
        base = float(self.compression_keep_ratio)
        if not self.compression_dynamic_enable or pose_summary is None:
            return base, None

        translation = float(pose_summary.get("translation_delta", 0.0))
        rotation = float(pose_summary.get("rotation_delta_rad", 0.0))
        motion_score = (
            translation / self.compression_dynamic_translation_scale
            + rotation / self.compression_dynamic_rotation_scale
        )
        keep_ratio = base - self.compression_dynamic_motion_weight * motion_score
        keep_ratio = min(self.compression_dynamic_max_keep, max(self.compression_dynamic_min_keep, keep_ratio))
        return keep_ratio, motion_score

    def append_block(
        self,
        *,
        block_id: int,
        frame_start: int,
        frame_count: int,
        frame_seq_length: int,
        cache_branches: dict[str, tuple[list[dict], Optional[list[dict]]]],
        viewmats=None,
        Ks=None,
    ) -> bool:
        if not self.enabled:
            return False
        if frame_count <= 0 or frame_seq_length <= 0:
            raise ValueError("frame_count and frame_seq_length must be positive")
        if not cache_branches:
            raise ValueError("cache_branches must not be empty")

        token_count = int(frame_count) * int(frame_seq_length)
        first_cache = next(iter(cache_branches.values()))[0]
        if not first_cache:
            raise ValueError("normal KV cache must not be empty")
        storage_device = self._storage_device(first_cache[0]["k"].device)
        if self.max_blocks > 0 and len(self.blocks) >= self.max_blocks:
            removed = self.blocks.pop(0)
            self.total_bytes -= removed.storage_bytes
            self.evicted_blocks += 1
        stored_branches = {}
        block_bytes = 0
        stored_viewmats, stored_Ks, pose_summary = _camera_metadata(viewmats, Ks)
        effective_keep_ratio, motion_score = self._effective_keep_ratio(pose_summary)

        for branch_name, (normal_cache, prope_cache) in cache_branches.items():
            if prope_cache is not None and len(prope_cache) != len(normal_cache):
                raise ValueError(
                    f"branch={branch_name}: normal/prope layer count mismatch "
                    f"({len(normal_cache)} != {len(prope_cache)})"
                )
            layers = []
            for layer_index, normal_entry in enumerate(normal_cache):
                k_source, v_source = _latest_cache_slice(normal_entry, token_count)
                if self.compression_enable and self.compression_at_store:
                    k_source, v_source = compress_retrieval_kv(
                        k_source,
                        v_source,
                        chunk_size=frame_count,
                        frame_seq_length=frame_seq_length,
                        keep_ratio=effective_keep_ratio,
                        anchor_rotate=False,
                        pooled=self.compression_pooled,
                    )
                k = _copy_tensor(k_source, storage_device)
                v = _copy_tensor(v_source, storage_device)
                prope_k = None
                prope_v = None
                if prope_cache is not None:
                    prope_k_source, prope_v_source = _latest_cache_slice(
                        prope_cache[layer_index], token_count
                    )
                    if self.compression_enable and self.compression_at_store:
                        prope_k_source, prope_v_source = compress_retrieval_kv(
                            prope_k_source,
                            prope_v_source,
                            chunk_size=frame_count,
                            frame_seq_length=frame_seq_length,
                            keep_ratio=effective_keep_ratio,
                            anchor_rotate=False,
                            pooled=self.compression_pooled,
                        )
                    prope_k = _copy_tensor(prope_k_source, storage_device)
                    prope_v = _copy_tensor(prope_v_source, storage_device)
                entry = KVBankLayerEntry(k=k, v=v, prope_k=prope_k, prope_v=prope_v)
                block_bytes += sum(
                    _tensor_bytes(tensor)
                    for tensor in (entry.k, entry.v, entry.prope_k, entry.prope_v)
                )
                layers.append(entry)
            stored_branches[branch_name] = layers

        block_bytes += _tensor_bytes(stored_viewmats) + _tensor_bytes(stored_Ks)
        block = KVBankBlock(
            block_id=int(block_id),
            frame_start=int(frame_start),
            frame_end=int(frame_start + frame_count),
            token_count=token_count,
            branches=stored_branches,
            viewmats=stored_viewmats,
            Ks=stored_Ks,
            pose_summary=pose_summary,
            storage_bytes=block_bytes,
            stored_compressed=self.compression_enable and self.compression_at_store,
            compression_keep_ratio=effective_keep_ratio if self.compression_enable and self.compression_at_store else None,
            motion_score=motion_score,
        )

        self.blocks.append(block)
        self.total_bytes += block_bytes

        self._log_append(block)
        return True

    def _log_append(self, block: KVBankBlock) -> None:
        if not is_main_process():
            return
        if not self._printed_projection:
            projected_blocks = self.expected_blocks
            if self.max_blocks > 0:
                projected_blocks = min(projected_blocks or self.max_blocks, self.max_blocks)
            projected_bytes = block.storage_bytes * projected_blocks
            print(
                f"[kv-bank-init] device={self.device_name} branches={list(block.branches)} "
                f"layers={len(next(iter(block.branches.values())))} "
                f"block_gb={block.storage_bytes / 1024 ** 3:.3f} "
                f"projected_blocks={projected_blocks} projected_gb={projected_bytes / 1024 ** 3:.3f}",
                flush=True,
            )
            if self.warn_memory_bytes > 0 and projected_bytes > self.warn_memory_bytes:
                print(
                    f"[kv-bank-warning] projected storage exceeds "
                    f"{self.warn_memory_bytes / 1024 ** 3:.1f} GB; set "
                    f"KV_BANK_MAX_BLOCKS to bound memory.",
                    flush=True,
                )
            self._printed_projection = True
        if block.block_id % self.log_interval == 0:
            print(
                f"[kv-bank] block={block.block_id} frames={block.frame_start}:{block.frame_end} "
                f"tokens={block.token_count} blocks={len(self.blocks)} "
                f"total_gb={self.total_bytes / 1024 ** 3:.3f} evicted={self.evicted_blocks} "
                f"keep_ratio={block.compression_keep_ratio if block.compression_keep_ratio is not None else 'none'} "
                f"motion_score={block.motion_score if block.motion_score is not None else 'none'}",
                flush=True,
            )

    def get_layer(
        self,
        block_index: int,
        layer_index: int,
        *,
        branch: str = "main",
    ) -> KVBankLayerEntry:
        return self.blocks[block_index].branches[branch][layer_index]

    def get_retrieval_payload(
        self,
        selected_block_indices: list[int],
        layer_index: int,
        *,
        branch: str = "main",
        device: torch.device | str | None = None,
        include_prope: bool = False,
        compress_runtime: bool = False,
        compression_keep_ratio: float = 0.5,
        compression_anchor_rotate: bool = False,
        compression_pooled: bool = False,
        frame_seq_length: int = 1560,
        rope_correction: bool = False,
        prope_reencode_mode: str = "none",
        current_viewmats: Optional[torch.Tensor] = None,
        current_Ks: Optional[torch.Tensor] = None,
    ) -> Optional[dict[str, torch.Tensor | list[int] | bool | int]]:
        if not selected_block_indices:
            return None

        device = torch.device(device) if device is not None else None
        entries = [self.blocks[index].branches[branch][layer_index] for index in selected_block_indices]
        k_parts = [entry.k.to(device=device) if device is not None else entry.k for entry in entries]
        v_parts = [entry.v.to(device=device) if device is not None else entry.v for entry in entries]
        payload = {
            "k": torch.cat(k_parts, dim=1),
            "v": torch.cat(v_parts, dim=1),
            "block_ids": [self.blocks[index].block_id for index in selected_block_indices],
            "src_frame_ids": [self.blocks[index].frame_start for index in selected_block_indices],
            "chunk_size_frames": self.blocks[selected_block_indices[0]].frame_end - self.blocks[selected_block_indices[0]].frame_start,
            "compress_chunk_size": self.blocks[selected_block_indices[0]].frame_end - self.blocks[selected_block_indices[0]].frame_start,
            "stored_compressed": all(self.blocks[index].stored_compressed for index in selected_block_indices),
            "rope_correction": bool(rope_correction),
            "prope_reencode_mode": normalize_prope_reencode_mode(prope_reencode_mode),
            "prope_reencoded": False,
        }

        if include_prope and all(entry.prope_k is not None and entry.prope_v is not None for entry in entries):
            prope_k_parts = []
            prope_v_parts = []
            reencoded_any = False
            mode = normalize_prope_reencode_mode(prope_reencode_mode)
            for block_index, entry in zip(selected_block_indices, entries):
                prope_k = entry.prope_k.to(device=device) if device is not None else entry.prope_k
                prope_v = entry.prope_v.to(device=device) if device is not None else entry.prope_v
                block = self.blocks[block_index]
                if mode == "current" and not block.stored_compressed:
                    prope_k, prope_v, did_reencode = reencode_prope_kv_to_current(
                        prope_k,
                        prope_v,
                        src_viewmats=block.viewmats,
                        src_Ks=block.Ks,
                        dst_viewmats=current_viewmats,
                        dst_Ks=current_Ks,
                    )
                    reencoded_any = reencoded_any or did_reencode
                prope_k_parts.append(prope_k)
                prope_v_parts.append(prope_v)
            payload["prope_k"] = torch.cat(prope_k_parts, dim=1)
            payload["prope_v"] = torch.cat(prope_v_parts, dim=1)
            payload["prope_reencoded"] = reencoded_any

        if compress_runtime and not payload["stored_compressed"]:
            chunk_size = int(payload["chunk_size_frames"])
            payload["k"], payload["v"] = compress_retrieval_kv(
                payload["k"],
                payload["v"],
                chunk_size=chunk_size,
                frame_seq_length=frame_seq_length,
                keep_ratio=compression_keep_ratio,
                anchor_rotate=compression_anchor_rotate,
                pooled=compression_pooled,
            )
            if "prope_k" in payload:
                payload["prope_k"], payload["prope_v"] = compress_retrieval_kv(
                    payload["prope_k"],
                    payload["prope_v"],
                    chunk_size=chunk_size,
                    frame_seq_length=frame_seq_length,
                    keep_ratio=compression_keep_ratio,
                    anchor_rotate=compression_anchor_rotate,
                    pooled=compression_pooled,
                )

        return payload

    def select_retrieval_blocks(
        self,
        *,
        current_viewmats: Optional[torch.Tensor],
        current_frame_start: int,
        current_frame_count: int,
        retrieval_frames: int,
        metric: str = "pose",
        fov_samples: int = 8192,
        fov_radius: float = 8.0,
        fov_h_deg: float = 60.0,
        fov_v_deg: float = 35.0,
        hybrid_fov_weight: float = 0.5,
        recent_frames: int = 0,
        sink_size_frames: int = 0,
        device: torch.device | str | None = None,
        return_details: bool = False,
    ) -> list[int] | tuple[list[int], dict[str, object]]:
        """Return bank-list indices selected for the current block."""
        if not self.enabled or not self.blocks or retrieval_frames <= 0:
            details = {"candidate_indices": [], "selected_indices": [], "distances": []}
            return ([], details) if return_details else []

        metric = normalize_retrieval_metric(metric)
        if metric == "recent_only":
            details = {"candidate_indices": [], "selected_indices": [], "distances": []}
            return ([], details) if return_details else []

        candidates = []
        for index, block in enumerate(self.blocks):
            if block.frame_start < sink_size_frames:
                continue
            if recent_frames > 0 and block.frame_end > current_frame_start - recent_frames:
                continue
            if block.frame_end > current_frame_start:
                continue
            if block.viewmats is None:
                continue
            candidates.append(index)

        if not candidates or current_viewmats is None:
            details = {"candidate_indices": candidates, "selected_indices": [], "distances": []}
            return ([], details) if return_details else []

        current_viewmats = current_viewmats.detach().float().cpu()
        candidate_viewmats = [self.blocks[index].viewmats for index in candidates]
        points = None
        if metric in {"worldkv_fov", "hy_fov", "hybrid"}:
            points = generate_deterministic_points_in_sphere(
                int(fov_samples),
                float(fov_radius),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        distances = retrieval_block_distances(
            current_viewmats,
            candidate_viewmats,
            metric=metric,
            points=points,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            fov_radius=fov_radius,
            hybrid_fov_weight=hybrid_fov_weight,
        )

        frames_per_block = max(1, int(current_frame_count))
        num_blocks = max(1, int(math.ceil(float(retrieval_frames) / frames_per_block)))
        selected = select_topk_indices(candidates, distances, num_blocks)
        details = {
            "candidate_indices": candidates,
            "candidate_block_ids": [self.blocks[index].block_id for index in candidates],
            "candidate_frame_starts": [self.blocks[index].frame_start for index in candidates],
            "selected_indices": selected,
            "selected_block_ids": [self.blocks[index].block_id for index in selected],
            "selected_frame_starts": [self.blocks[index].frame_start for index in selected],
            "distances": distances.detach().float().cpu().tolist(),
        }
        return (selected, details) if return_details else selected

    def get_retrieval_payloads(
        self,
        selected_block_indices: list[int],
        *,
        branch: str,
        device: torch.device | str | None,
        include_prope: bool,
        compress_runtime: bool,
        compression_keep_ratio: float,
        compression_anchor_rotate: bool,
        compression_pooled: bool,
        frame_seq_length: int,
        rope_correction: bool = False,
        prope_reencode_mode: str = "none",
        current_viewmats: Optional[torch.Tensor] = None,
        current_Ks: Optional[torch.Tensor] = None,
    ) -> Optional[list[dict[str, torch.Tensor | list[int] | bool | int]]]:
        if not selected_block_indices:
            return None
        first_block = self.blocks[selected_block_indices[0]]
        if branch not in first_block.branches:
            return None
        num_layers = len(first_block.branches[branch])
        payloads = []
        for layer_index in range(num_layers):
            payload = self.get_retrieval_payload(
                selected_block_indices,
                layer_index,
                branch=branch,
                device=device,
                include_prope=include_prope,
                compress_runtime=compress_runtime,
                compression_keep_ratio=compression_keep_ratio,
                compression_anchor_rotate=compression_anchor_rotate,
                compression_pooled=compression_pooled,
                frame_seq_length=frame_seq_length,
                rope_correction=rope_correction,
                prope_reencode_mode=prope_reencode_mode,
                current_viewmats=current_viewmats,
                current_Ks=current_Ks,
            )
            if payload is None:
                return None
            payloads.append(payload)
        return payloads

    def summary(self) -> dict[str, object]:
        branches = sorted(self.blocks[0].branches) if self.blocks else []
        return {
            "enabled": self.enabled,
            "device": self.device_name,
            "blocks": len(self.blocks),
            "evicted_blocks": self.evicted_blocks,
            "total_bytes": self.total_bytes,
            "total_gb": self.total_bytes / 1024 ** 3,
            "max_blocks": self.max_blocks,
            "branches": branches,
            "compression_enable": self.compression_enable,
            "compression_at_store": self.compression_at_store,
            "compression_keep_ratio": self.compression_keep_ratio,
            "compression_pooled": self.compression_pooled,
            "compression_dynamic_enable": self.compression_dynamic_enable,
            "compression_dynamic_min_keep": self.compression_dynamic_min_keep,
            "compression_dynamic_max_keep": self.compression_dynamic_max_keep,
            "compression_dynamic_translation_scale": self.compression_dynamic_translation_scale,
            "compression_dynamic_rotation_scale": self.compression_dynamic_rotation_scale,
            "compression_dynamic_motion_weight": self.compression_dynamic_motion_weight,
        }
