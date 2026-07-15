import os

import torch
import torch.distributed as dist


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def should_log_cache(enabled: bool, block_index: int, interval: int) -> bool:
    if not enabled or not is_main_process():
        return False
    interval = max(1, int(interval))
    return block_index % interval == 0


def cache_index_summary(cache):
    if not cache:
        return "none"
    entry = cache[0]
    local_end = int(entry["local_end_index"].item())
    global_end = int(entry["global_end_index"].item())
    capacity = int(entry["k"].shape[1])
    return f"local_end={local_end} global_end={global_end} capacity={capacity}"


def cuda_memory_summary(device) -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    device = torch.device(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"mem_alloc={allocated:.2f}GB mem_reserved={reserved:.2f}GB mem_peak={peak:.2f}GB"


def pose_summary(viewmats) -> str:
    if viewmats is None:
        return "pose=none"
    with torch.no_grad():
        vm = viewmats.detach().float()
        trans = vm[..., :3, 3]
        trans_mean = trans.mean(dim=tuple(range(trans.ndim - 1)))
        trans_norm = torch.linalg.norm(trans_mean).item()
        frames = int(vm.shape[1]) if vm.ndim >= 4 else int(vm.shape[0])
    return f"pose_frames={frames} pose_t_norm={trans_norm:.4f}"


def log_cache_state(
    *,
    tag: str,
    block_index: int,
    current_start_frame: int,
    current_num_frames: int,
    frame_seq_length: int,
    kv_cache=None,
    prope_kv_cache=None,
    viewmats=None,
    device=None,
) -> None:
    if not is_main_process():
        return
    device = device or os.environ.get("CUDA_VISIBLE_DEVICES", "cuda")
    print(
        "[cache-state] "
        f"tag={tag} "
        f"block={block_index} "
        f"frames={current_start_frame}:{current_start_frame + current_num_frames} "
        f"tokens={current_start_frame * frame_seq_length}:{(current_start_frame + current_num_frames) * frame_seq_length} "
        f"kv=({cache_index_summary(kv_cache)}) "
        f"prope=({cache_index_summary(prope_kv_cache)}) "
        f"{pose_summary(viewmats)} "
        f"{cuda_memory_summary(device)}",
        flush=True,
    )
