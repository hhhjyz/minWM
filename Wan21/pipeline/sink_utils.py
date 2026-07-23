"""Utilities for sink-frame KV cache experiments.

Periodic sink v0 directly copies already-encoded RoPE/PRoPE KV tensors into the
sink slots. It does not recompute RoPE or PRoPE for the new cache positions.
This is useful as a low-cost ablation, but it is not a geometrically strict
position-corrected replay mechanism.
"""

from __future__ import annotations

import random

from .kv_bank import normalize_prope_reencode_mode, reencode_prope_kv_to_current


BANK_SINK_STRATEGIES = {
    "bank_random",
    "bank_uniform",
    "bank_pose",
    "bank_worldkv_fov",
}


def get_model_sink_size_frames(generator) -> int:
    try:
        block = generator.model.blocks[0]
        return int(block.self_attn.sink_size)
    except Exception:
        return 0


def normalize_sink_strategy(strategy: str) -> str:
    strategy = (strategy or "none").lower()
    if strategy == "bank_fov":
        strategy = "bank_worldkv_fov"
    valid = {"none", "fixed", "periodic"} | BANK_SINK_STRATEGIES
    if strategy not in valid:
        raise ValueError(f"Unknown sink_strategy={strategy!r}; expected one of {sorted(valid)}")
    return strategy


def is_bank_sink_strategy(strategy: str) -> bool:
    return normalize_sink_strategy(strategy) in BANK_SINK_STRATEGIES


def copy_latest_tokens_to_sink(cache_list, sink_tokens: int, latest_tokens: int) -> int:
    """Copy the newest cached K/V tokens into the leading sink slots.

    Important: this copies the cached tensors as-is. For PRoPE caches, the
    tensors have already been transformed with the source block's camera pose.
    No RoPE/PRoPE re-encoding is performed after moving them to the sink area.
    """
    if not cache_list or sink_tokens <= 0 or latest_tokens <= 0:
        return 0

    copied = 0
    for cache in cache_list:
        local_end = int(cache["local_end_index"].item())
        copy_tokens = min(sink_tokens, latest_tokens, local_end)
        if copy_tokens <= 0:
            continue

        src_start = local_end - copy_tokens
        src_end = local_end
        cache["k"][:, :copy_tokens] = cache["k"][:, src_start:src_end].clone()
        cache["v"][:, :copy_tokens] = cache["v"][:, src_start:src_end].clone()
        copied = max(copied, copy_tokens)
    return copied


def _zero_stale_sink_tail(cache, copied_tokens: int, sink_tokens: int) -> None:
    if copied_tokens >= sink_tokens:
        return
    end = min(int(sink_tokens), int(cache["k"].shape[1]))
    if copied_tokens >= end:
        return
    cache["k"][:, copied_tokens:end].zero_()
    cache["v"][:, copied_tokens:end].zero_()


def copy_bank_block_to_sink(
    *,
    kv_cache,
    prope_kv_cache,
    bank_block,
    branch: str,
    sink_tokens: int,
    prope_reencode_mode: str = "none",
    current_viewmats=None,
    current_Ks=None,
) -> tuple[int, int]:
    """Copy a selected KV-bank block into the leading sink slots.

    The selected tensors are copied as stored. If the bank block was compressed
    and has fewer than ``sink_tokens``, the remaining sink slots are zeroed to
    avoid leaking the previous sink content.
    """
    if bank_block is None or branch not in bank_block.branches or sink_tokens <= 0:
        return 0, 0
    if not kv_cache:
        return 0, 0

    layers = bank_block.branches[branch]
    copied = 0
    for layer_index, cache in enumerate(kv_cache):
        if layer_index >= len(layers):
            break
        entry = layers[layer_index]
        copy_tokens = min(int(sink_tokens), int(entry.k.shape[1]), int(cache["k"].shape[1]))
        if copy_tokens <= 0:
            continue
        cache["k"][:, :copy_tokens] = entry.k[:, :copy_tokens].to(
            device=cache["k"].device,
            dtype=cache["k"].dtype,
        ).clone()
        cache["v"][:, :copy_tokens] = entry.v[:, :copy_tokens].to(
            device=cache["v"].device,
            dtype=cache["v"].dtype,
        ).clone()
        _zero_stale_sink_tail(cache, copy_tokens, sink_tokens)
        copied = max(copied, copy_tokens)

    prope_copied = 0
    prope_reencoded = False
    reencode_mode = normalize_prope_reencode_mode(prope_reencode_mode)
    if prope_kv_cache:
        for layer_index, cache in enumerate(prope_kv_cache):
            if layer_index >= len(layers):
                break
            entry = layers[layer_index]
            if entry.prope_k is None or entry.prope_v is None:
                _zero_stale_sink_tail(cache, 0, sink_tokens)
                continue
            prope_k = entry.prope_k
            prope_v = entry.prope_v
            if reencode_mode == "current" and not getattr(bank_block, "stored_compressed", False):
                prope_k, prope_v, did_reencode = reencode_prope_kv_to_current(
                    prope_k,
                    prope_v,
                    src_viewmats=bank_block.viewmats,
                    src_Ks=bank_block.Ks,
                    dst_viewmats=current_viewmats,
                    dst_Ks=current_Ks,
                )
                prope_reencoded = prope_reencoded or did_reencode
            copy_tokens = min(int(sink_tokens), int(prope_k.shape[1]), int(cache["k"].shape[1]))
            if copy_tokens <= 0:
                continue
            cache["k"][:, :copy_tokens] = prope_k[:, :copy_tokens].to(
                device=cache["k"].device,
                dtype=cache["k"].dtype,
            ).clone()
            cache["v"][:, :copy_tokens] = prope_v[:, :copy_tokens].to(
                device=cache["v"].device,
                dtype=cache["v"].dtype,
            ).clone()
            _zero_stale_sink_tail(cache, copy_tokens, sink_tokens)
            prope_copied = max(prope_copied, copy_tokens)

    return copied, prope_copied, prope_reencoded


def _bank_sink_candidates(
    *,
    kv_bank,
    branch: str,
    current_frame_start: int,
    recent_frames: int,
    sink_size_frames: int,
) -> list[int]:
    if kv_bank is None or not getattr(kv_bank, "enabled", False):
        return []
    candidates = []
    for index, block in enumerate(kv_bank.blocks):
        if branch not in block.branches:
            continue
        if block.frame_start < sink_size_frames:
            continue
        if recent_frames > 0 and block.frame_end > current_frame_start - recent_frames:
            continue
        if block.frame_end > current_frame_start:
            continue
        candidates.append(index)
    return candidates


def _select_bank_sink_block(
    *,
    strategy: str,
    kv_bank,
    branch: str,
    block_index: int,
    update_interval: int,
    current_frame_start: int,
    current_num_frames: int,
    current_viewmats,
    sink_size_frames: int,
    recent_frames: int,
    fov_samples: int,
    fov_radius: float,
    fov_h_deg: float,
    fov_v_deg: float,
    hybrid_fov_weight: float,
    random_seed: int,
    label: str,
) -> tuple[int | None, dict[str, object]]:
    details: dict[str, object] = {"candidate_indices": [], "selected_indices": []}
    if strategy in {"bank_pose", "bank_worldkv_fov"}:
        metric = "pose" if strategy == "bank_pose" else "worldkv_fov"
        selected, details = kv_bank.select_retrieval_blocks(
            current_viewmats=current_viewmats,
            current_frame_start=current_frame_start,
            current_frame_count=current_num_frames,
            retrieval_frames=sink_size_frames,
            metric=metric,
            fov_samples=fov_samples,
            fov_radius=fov_radius,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
            hybrid_fov_weight=hybrid_fov_weight,
            recent_frames=recent_frames,
            sink_size_frames=sink_size_frames,
            return_details=True,
        )
        selected = [index for index in selected if branch in kv_bank.blocks[index].branches]
        return (selected[0] if selected else None), details

    candidates = _bank_sink_candidates(
        kv_bank=kv_bank,
        branch=branch,
        current_frame_start=current_frame_start,
        recent_frames=recent_frames,
        sink_size_frames=sink_size_frames,
    )
    details["candidate_indices"] = candidates
    details["candidate_block_ids"] = [kv_bank.blocks[index].block_id for index in candidates]
    details["candidate_frame_starts"] = [kv_bank.blocks[index].frame_start for index in candidates]
    if not candidates:
        return None, details

    if strategy == "bank_random":
        label_offset = sum(ord(ch) for ch in str(label))
        rng = random.Random(int(random_seed) + int(block_index) * 1009 + label_offset)
        selected_index = rng.choice(candidates)
    else:
        update_index = max(0, (int(block_index) + 1) // max(1, int(update_interval)) - 1)
        selected_index = candidates[update_index % len(candidates)]

    details["selected_indices"] = [selected_index]
    details["selected_block_ids"] = [kv_bank.blocks[selected_index].block_id]
    details["selected_frame_starts"] = [kv_bank.blocks[selected_index].frame_start]
    return selected_index, details


def maybe_update_periodic_sink(
    *,
    strategy: str,
    update_interval: int,
    block_index: int,
    current_num_frames: int,
    frame_seq_length: int,
    sink_size_frames: int,
    kv_cache,
    prope_kv_cache=None,
    label: str = "",
) -> int:
    strategy = normalize_sink_strategy(strategy)
    if strategy != "periodic":
        return 0
    if sink_size_frames <= 0 or update_interval <= 0:
        return 0
    if (block_index + 1) % update_interval != 0:
        return 0

    sink_tokens = sink_size_frames * frame_seq_length
    latest_tokens = current_num_frames * frame_seq_length
    # v0 approximation: both normal KV and PRoPE KV are copied as encoded memory.
    # Future retrieval-quality implementations should store raw K/V + pose/frame
    # metadata and apply RoPE/PRoPE correction when replaying memory.
    copied = copy_latest_tokens_to_sink(kv_cache, sink_tokens, latest_tokens)
    prope_copied = copy_latest_tokens_to_sink(prope_kv_cache, sink_tokens, latest_tokens)
    print(
        f"[sink-update] strategy=periodic label={label} block={block_index} "
        f"interval={update_interval} sink_frames={sink_size_frames} "
        f"kv_tokens={copied} prope_tokens={prope_copied}"
    )
    return copied


def maybe_update_sink(
    *,
    strategy: str,
    update_interval: int,
    block_index: int,
    current_frame_start: int,
    current_num_frames: int,
    frame_seq_length: int,
    sink_size_frames: int,
    kv_cache,
    prope_kv_cache=None,
    kv_bank=None,
    branch: str = "main",
    current_viewmats=None,
    current_Ks=None,
    recent_frames: int = 0,
    fov_samples: int = 8192,
    fov_radius: float = 8.0,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 35.0,
    hybrid_fov_weight: float = 0.5,
    random_seed: int = 0,
    prope_reencode_mode: str = "none",
    label: str = "",
) -> int:
    strategy = normalize_sink_strategy(strategy)
    if strategy in {"none", "fixed"}:
        return 0
    if strategy == "periodic":
        return maybe_update_periodic_sink(
            strategy=strategy,
            update_interval=update_interval,
            block_index=block_index,
            current_num_frames=current_num_frames,
            frame_seq_length=frame_seq_length,
            sink_size_frames=sink_size_frames,
            kv_cache=kv_cache,
            prope_kv_cache=prope_kv_cache,
            label=label,
        )

    if sink_size_frames <= 0 or update_interval <= 0:
        return 0
    if (block_index + 1) % update_interval != 0:
        return 0
    if kv_bank is None or not getattr(kv_bank, "enabled", False):
        print(
            f"[sink-update] strategy={strategy} label={label} block={block_index} "
            "skipped=no_kv_bank",
            flush=True,
        )
        return 0

    selected_index, details = _select_bank_sink_block(
        strategy=strategy,
        kv_bank=kv_bank,
        branch=branch,
        block_index=block_index,
        update_interval=update_interval,
        current_frame_start=current_frame_start,
        current_num_frames=current_num_frames,
        current_viewmats=current_viewmats,
        sink_size_frames=sink_size_frames,
        recent_frames=recent_frames,
        fov_samples=fov_samples,
        fov_radius=fov_radius,
        fov_h_deg=fov_h_deg,
        fov_v_deg=fov_v_deg,
        hybrid_fov_weight=hybrid_fov_weight,
        random_seed=random_seed,
        label=label,
    )
    if selected_index is None:
        print(
            f"[sink-update] strategy={strategy} label={label} block={block_index} "
            f"skipped=no_candidate candidates={details.get('candidate_block_ids', [])}",
            flush=True,
        )
        return 0

    sink_tokens = sink_size_frames * frame_seq_length
    selected_block = kv_bank.blocks[selected_index]
    copied, prope_copied, prope_reencoded = copy_bank_block_to_sink(
        kv_cache=kv_cache,
        prope_kv_cache=prope_kv_cache,
        bank_block=selected_block,
        branch=branch,
        sink_tokens=sink_tokens,
        prope_reencode_mode=prope_reencode_mode,
        current_viewmats=current_viewmats,
        current_Ks=current_Ks,
    )
    print(
        f"[sink-update] strategy={strategy} label={label} branch={branch} block={block_index} "
        f"interval={update_interval} sink_frames={sink_size_frames} "
        f"selected_block={selected_block.block_id} selected_frames={selected_block.frame_start}:{selected_block.frame_end} "
        f"stored_compressed={selected_block.stored_compressed} kv_tokens={copied} "
        f"prope_tokens={prope_copied} prope_reencode={normalize_prope_reencode_mode(prope_reencode_mode)} "
        f"prope_reencoded={int(prope_reencoded)}",
        flush=True,
    )
    return copied
