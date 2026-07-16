"""Utilities for sink-frame KV cache experiments.

Periodic sink v0 directly copies already-encoded RoPE/PRoPE KV tensors into the
sink slots. It does not recompute RoPE or PRoPE for the new cache positions.
This is useful as a low-cost ablation, but it is not a geometrically strict
position-corrected replay mechanism.
"""


def get_model_sink_size_frames(generator) -> int:
    try:
        block = generator.model.blocks[0]
        return int(block.self_attn.sink_size)
    except Exception:
        return 0


def normalize_sink_strategy(strategy: str) -> str:
    strategy = (strategy or "none").lower()
    if strategy not in {"none", "fixed", "periodic"}:
        raise ValueError(f"Unknown sink_strategy={strategy!r}; expected none/fixed/periodic")
    return strategy


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
