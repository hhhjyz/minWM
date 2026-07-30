import argparse
import csv
import hashlib
import torch
import os
import time
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
import json

from wan_utils.dataset import TextDataset, TextImagePairDataset
from wan_utils.misc import set_seed
from wan_utils.camera_trajectory import make_camera_tensors

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller


def safe_video_filename(prompt: str, camera_suffix: str, *, max_length: int = 180) -> str:
    raw = f"{prompt[:100]}{camera_suffix}"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    safe = "_".join(part for part in safe.split("_") if part)
    if not safe:
        safe = "video"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    suffix = f"_{digest}"
    if len(safe) + len(suffix) > max_length:
        safe = safe[: max(1, max_length - len(suffix))].rstrip("._-")
    return f"{safe}{suffix}.mp4"


def summarize_camera_pose(viewmats):
    if viewmats is None:
        return {
            "camera_pose_frames": 0,
            "camera_translation_range": "",
            "camera_translation_final": "",
            "camera_rotation_range_deg": "",
            "camera_rotation_final_deg": "",
            "camera_first_motion_frame": "",
        }

    with torch.no_grad():
        c2w = torch.linalg.inv(viewmats.detach().float().cpu()[0])
        positions = c2w[:, :3, 3]
        rotations = c2w[:, :3, :3]
        rel = rotations[0].transpose(-1, -2).unsqueeze(0) @ rotations
        trace = rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        angles = torch.rad2deg(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
        trans_from_first = torch.linalg.norm(positions - positions[:1], dim=-1)
        if len(c2w) > 1:
            step_trans = torch.linalg.norm(positions[1:] - positions[:-1], dim=-1)
            step_rel = rotations[:-1].transpose(-1, -2) @ rotations[1:]
            step_trace = step_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            step_angles = torch.rad2deg(torch.acos(((step_trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
            motion = step_trans + step_angles / 30.0
            moving = torch.nonzero(motion > 1e-3, as_tuple=False)
            first_motion_frame = int(moving[0].item() + 1) if moving.numel() else ""
        else:
            first_motion_frame = ""

    return {
        "camera_pose_frames": int(len(c2w)),
        "camera_translation_range": float(trans_from_first.max().item()),
        "camera_translation_final": float(trans_from_first[-1].item()),
        "camera_rotation_range_deg": float(angles.max().item()),
        "camera_rotation_final_deg": float(angles[-1].item()),
        "camera_first_motion_frame": first_motion_frame,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--output_folder", type=str, help="Output folder")
parser.add_argument("--num_output_frames", type=int, default=20, help="Number of overlap frames between sliding windows")
parser.add_argument("--max_prompts", type=int, default=-1, help="Maximum number of prompts to run. <=0 means all prompts.")
parser.add_argument("--prompt_start", type=int, default=0, help="Start prompt index for subset inference.")
parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
parser.add_argument("--seed", type=int, default=0, help="Random seed")
parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
parser.add_argument("--sp_size", type=int, default=1, help="Sequence parallel size (1=disabled)")
parser.add_argument("--trajectory", type=str, default=None, help="Camera trajectory string (e.g., 'w*19' for camera control)")
parser.add_argument("--trajectory_path", type=str, default=None, help="Path to trajectory file (one trajectory string per line, aligned with data_path)")
parser.add_argument("--log_cache_state", action="store_true", help="Log per-block KV/PRoPE cache state and CUDA memory.")
parser.add_argument("--log_cache_interval", type=int, default=1, help="Log every N generated blocks when --log_cache_state is enabled.")
parser.add_argument("--sink_strategy", type=str, default="none", choices=["none", "fixed", "periodic", "bank_random", "bank_uniform", "bank_pose", "bank_worldkv_fov", "bank_fov"], help="Sink-frame KV cache strategy.")
parser.add_argument("--sink_size", type=int, default=0, help="Number of latent frames reserved as sink frames.")
parser.add_argument("--fixed_sink_rope_rebase", action="store_true", help="Deprecated alias for --tri_region_rope_rebase.")
parser.add_argument("--tri_region_rope_rebase", action="store_true", help="Map sink, retrieval memory, recent K, and current Q/K into bounded virtual RoPE regions.")
parser.add_argument("--rope_train_length", type=int, default=21, help="Virtual RoPE position of the final current query frame.")
parser.add_argument("--rope_local_window", type=int, default=9, help="Number of recent frames placed immediately before the current query region.")
parser.add_argument("--sink_update_interval", type=int, default=0, help="For periodic/bank sink, update sink every N generated blocks.")
parser.add_argument("--sink_bank_seed", type=int, default=0, help="Deterministic seed for bank_random sink selection.")
parser.add_argument("--kv_bank_enable", action="store_true", help="Store clean per-block KV in a sidecar bank without retrieval.")
parser.add_argument("--kv_bank_device", choices=["cpu", "cuda"], default="cpu", help="Storage device for KV bank tensors.")
parser.add_argument("--kv_bank_max_blocks", type=int, default=0, help="Maximum retained bank blocks. 0 means unlimited; positive values evict oldest blocks.")
parser.add_argument("--kv_bank_log_interval", type=int, default=1, help="Log KV bank state every N generated blocks.")
parser.add_argument("--kv_bank_warn_memory_gb", type=float, default=16.0, help="Warn when projected KV bank storage exceeds this many GB. 0 disables warning.")
parser.add_argument("--retrieval_enable", action="store_true", help="Enable WorldKV-style [sink | retrieved | recent] attention window.")
parser.add_argument("--retrieval_granularity", choices=["chunk", "latent_frame"], default="chunk", help="Select whole generated chunks or individual latent frames from the KV bank.")
parser.add_argument("--retrieval_metric", choices=["recent_only", "pose", "worldkv_fov", "hy_fov", "hybrid", "fov"], default="pose", help="Retrieval metric: WorldKV pose, WorldKV FOV, HY-WorldPlay FOV, or hybrid.")
parser.add_argument("--retrieval_frames", type=int, default=0, help="Number of latent frames to retrieve from KV bank.")
parser.add_argument("--retrieval_recent_frames", type=int, default=0, help="Exclude bank blocks that overlap the most recent N latent frames before current block. 0 disables this exclusion.")
parser.add_argument("--retrieval_fov_samples", type=int, default=8192, help="Number of deterministic probe points for FOV retrieval.")
parser.add_argument("--retrieval_fov_radius", type=float, default=8.0, help="Probe radius used by FOV retrieval.")
parser.add_argument("--retrieval_fov_h_deg", type=float, default=60.0, help="Horizontal FOV in degrees for retrieval scoring.")
parser.add_argument("--retrieval_fov_v_deg", type=float, default=35.0, help="Vertical FOV in degrees for retrieval scoring.")
parser.add_argument("--retrieval_hybrid_fov_weight", type=float, default=0.5, help="FOV distance weight for hybrid retrieval.")
parser.add_argument("--retrieval_rope_correction", action="store_true", help="WorldKV-style time-axis RoPE rebasing for retrieved normal KV.")
parser.add_argument("--prope_reencode_mode", choices=["none", "current"], default="none", help="Experimental PRoPE KV re-encoding for retrieved/bank-sink memory.")
parser.add_argument("--kv_compression_enable", action="store_true", help="Enable WorldKV-style anchor + novelty KV compression.")
parser.add_argument("--kv_compression_keep_ratio", type=float, default=0.5, help="Token keep ratio for non-anchor frames.")
parser.add_argument("--kv_compression_anchor_rotate", action="store_true", help="Rotate the anchor frame during runtime retrieval compression.")
parser.add_argument("--kv_compression_at_store", action="store_true", help="Compress each block once when storing into KV bank.")
parser.add_argument("--kv_compression_pooled", action="store_true", help="Pool the keep budget across non-anchor frames during KV compression.")
parser.add_argument("--kv_compression_dynamic_enable", action="store_true", help="Adjust store-time KV compression keep ratio from camera motion amplitude.")
parser.add_argument("--kv_compression_dynamic_min_keep", type=float, default=0.25, help="Minimum dynamic KV compression keep ratio.")
parser.add_argument("--kv_compression_dynamic_max_keep", type=float, default=0.75, help="Maximum dynamic KV compression keep ratio.")
parser.add_argument("--kv_compression_dynamic_translation_scale", type=float, default=1.0, help="Translation delta that maps to one dynamic compression motion unit.")
parser.add_argument("--kv_compression_dynamic_rotation_scale", type=float, default=0.35, help="Rotation delta in radians that maps to one dynamic compression motion unit.")
parser.add_argument("--kv_compression_dynamic_motion_weight", type=float, default=0.25, help="Keep-ratio reduction per dynamic compression motion unit.")
args = parser.parse_args()

# Initialize distributed inference
# IMPORTANT: distributed init MUST happen before importing pipeline modules,
# because causal_model.py checks for CleanCode SP infra at import time.
if args.sp_size > 1:
    # SP mode requires torchrun with nproc_per_node >= sp_size
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size_env >= args.sp_size, (
        f"SP requires at least {args.sp_size} processes, but WORLD_SIZE={world_size_env}. "
        f"Launch with: torchrun --nproc_per_node={args.sp_size} inference.py ... --sp_size {args.sp_size}"
    )
    from wan_utils.distributed import launch_distributed_job, get_sp_seed_offset
    launch_distributed_job(backend="nccl", sp_size=args.sp_size)
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
elif "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1

# Seed: under SP, ranks in the same SP group must share the same seed
if args.sp_size > 1:
    set_seed(args.seed + get_sp_seed_offset())
else:
    set_seed(args.seed)

# Refresh gpu device handle (demo_utils.memory.gpu is captured at import time
# before distributed init sets the correct CUDA device)
from demo_utils import memory as _mem
_mem.gpu = torch.device(f'cuda:{local_rank}')
gpu = _mem.gpu

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("Wan21/configs/default_config.yaml")
config = OmegaConf.merge(default_config, config)
config.log_cache_state = args.log_cache_state
config.log_cache_interval = max(1, args.log_cache_interval)
if args.sink_strategy != "none":
    assert args.sink_size > 0, "--sink_size must be > 0 when sink_strategy is enabled"
tri_region_rope_rebase = bool(
    args.tri_region_rope_rebase or args.fixed_sink_rope_rebase
)
if tri_region_rope_rebase:
    assert args.sink_strategy == "fixed", "tri-region RoPE requires --sink_strategy fixed"
    assert args.rope_train_length > 0, "--rope_train_length must be positive"
    assert args.rope_local_window >= 0, "--rope_local_window must be non-negative"
    assert args.sink_size < args.rope_train_length - args.rope_local_window, (
        "tri-region RoPE requires room between sink and local regions"
    )
if args.sink_strategy in {"periodic", "bank_random", "bank_uniform", "bank_pose", "bank_worldkv_fov", "bank_fov"}:
    assert args.sink_update_interval > 0, "--sink_update_interval must be > 0 for updating sink strategies"
model_kwargs = config.get("model_kwargs", {})
local_attn_size = int(model_kwargs.get("local_attn_size", -1))
if args.sink_size > 0 and local_attn_size != -1:
    assert args.sink_size < local_attn_size, (
        f"sink_size={args.sink_size} must be smaller than local_attn_size={local_attn_size}"
    )
effective_sink_size = int(args.sink_size) if args.sink_strategy != "none" else 0
config.model_kwargs.sink_size = effective_sink_size
config.model_kwargs.fixed_sink_rope_rebase = False
config.model_kwargs.tri_region_rope_rebase = tri_region_rope_rebase
config.model_kwargs.rope_train_length = int(args.rope_train_length)
config.model_kwargs.rope_local_window = int(args.rope_local_window)
config.sink_strategy = args.sink_strategy
config.sink_update_interval = int(args.sink_update_interval)
config.sink_bank_seed = int(args.sink_bank_seed)
config.kv_bank_enable = bool(args.kv_bank_enable)
config.kv_bank_device = args.kv_bank_device
config.kv_bank_max_blocks = max(0, int(args.kv_bank_max_blocks))
config.kv_bank_log_interval = max(1, int(args.kv_bank_log_interval))
config.kv_bank_warn_memory_gb = max(0.0, float(args.kv_bank_warn_memory_gb))
config.retrieval_enable = bool(args.retrieval_enable)
config.retrieval_granularity = args.retrieval_granularity
config.retrieval_metric = args.retrieval_metric
config.retrieval_frames = max(0, int(args.retrieval_frames))
config.retrieval_recent_frames = max(0, int(args.retrieval_recent_frames))
config.retrieval_fov_samples = max(1, int(args.retrieval_fov_samples))
config.retrieval_fov_radius = float(args.retrieval_fov_radius)
config.retrieval_fov_h_deg = float(args.retrieval_fov_h_deg)
config.retrieval_fov_v_deg = float(args.retrieval_fov_v_deg)
config.retrieval_hybrid_fov_weight = float(args.retrieval_hybrid_fov_weight)
config.retrieval_rope_correction = bool(args.retrieval_rope_correction)
config.prope_reencode_mode = args.prope_reencode_mode
config.kv_compression_enable = bool(args.kv_compression_enable)
config.kv_compression_keep_ratio = float(args.kv_compression_keep_ratio)
config.kv_compression_anchor_rotate = bool(args.kv_compression_anchor_rotate)
config.kv_compression_at_store = bool(args.kv_compression_at_store)
config.kv_compression_pooled = bool(args.kv_compression_pooled)
config.kv_compression_dynamic_enable = bool(args.kv_compression_dynamic_enable)
config.kv_compression_dynamic_min_keep = float(args.kv_compression_dynamic_min_keep)
config.kv_compression_dynamic_max_keep = float(args.kv_compression_dynamic_max_keep)
config.kv_compression_dynamic_translation_scale = float(args.kv_compression_dynamic_translation_scale)
config.kv_compression_dynamic_rotation_scale = float(args.kv_compression_dynamic_rotation_scale)
config.kv_compression_dynamic_motion_weight = float(args.kv_compression_dynamic_motion_weight)

# Import pipeline AFTER distributed init so causal_model.py sees CleanCode SP infra
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
    BidirectionalDiffusionInferencePipeline,
    BidirectionalInferencePipeline,
)

# Initialize pipeline
is_causal = config.get('causal', True)

if hasattr(config, 'denoising_step_list'):
    # Few-step inference
    if is_causal:
        pipeline = CausalInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalInferencePipeline(config, device=device)
else:
    # Multi-step diffusion inference
    if is_causal:
        pipeline = CausalDiffusionInferencePipeline(config, device=device)
    else:
        pipeline = BidirectionalDiffusionInferencePipeline(config, device=device)

if args.checkpoint_path:
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    key = 'generator_ema'
    try:
        gen_sd = state_dict[key]
    except:
        key = 'generator'
        gen_sd = state_dict[key]
    
    try:
        pipeline.generator.load_state_dict(gen_sd)
    except RuntimeError:
        fixed = {}
        for k, v in gen_sd.items():
            if k.startswith("model._fsdp_wrapped_module."):
                k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
            fixed[k] = v
        pipeline.generator.load_state_dict(fixed, strict=False)

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.generator.to(device=gpu)
pipeline.vae.to(device=gpu)


# Create dataset
if args.i2v:
    assert not dist.is_initialized(), "I2V does not support distributed inference yet"
    transform = transforms.Compose([
        transforms.Resize((480, 832)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = TextImagePairDataset(args.data_path, transform=transform)
else:
    dataset = TextDataset(prompt_path=args.data_path)
original_num_prompts = len(dataset)
assert args.prompt_start >= 0, f"prompt_start must be >= 0, got {args.prompt_start}"
assert args.prompt_start < original_num_prompts, (
    f"prompt_start={args.prompt_start} is out of range for {original_num_prompts} prompts"
)
if args.max_prompts > 0:
    prompt_end = min(original_num_prompts, args.prompt_start + args.max_prompts)
else:
    prompt_end = original_num_prompts
selected_indices = list(range(args.prompt_start, prompt_end))
assert selected_indices, (
    f"empty prompt subset: prompt_start={args.prompt_start}, "
    f"max_prompts={args.max_prompts}, total={original_num_prompts}"
)
if len(selected_indices) != original_num_prompts or args.prompt_start != 0:
    dataset = Subset(dataset, selected_indices)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts} / {original_num_prompts} (start={args.prompt_start})")

if dist.is_initialized() and args.sp_size <= 1:
    # Standard DP: split prompts across ranks
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
elif dist.is_initialized() and args.sp_size > 1:
    # SP mode: use SP-aware sampler so ranks in the same SP group get the same data
    from wan_utils.distributed import get_sp_data_sampler
    sampler = get_sp_data_sampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

timing_csv_path = os.path.join(args.output_folder, "inference_times.csv")
timing_json_path = os.path.join(args.output_folder, "inference_times.json")
retrieval_jsonl_path = os.path.join(args.output_folder, "retrieval_events.jsonl")
retrieval_csv_path = os.path.join(args.output_folder, "retrieval_events.csv")
timing_rows = []
retrieval_event_rows = []
if local_rank == 0:
    with open(timing_csv_path, "w", newline="", encoding="utf-8") as _f:
        _writer = csv.DictWriter(
            _f,
            fieldnames=[
                "sample_order",
                "prompt_index",
                "status",
                "prompt",
                "trajectory",
                "num_output_frames",
                "camera_pose_frames",
                "camera_translation_range",
                "camera_translation_final",
                "camera_rotation_range_deg",
                "camera_rotation_final_deg",
                "camera_first_motion_frame",
                "num_generated_latent_frames",
                "generation_seconds",
                "postprocess_seconds",
                "write_video_seconds",
                "total_seconds",
                "chunk0_latency_seconds",
                "last_chunk_seconds",
                "last_chunk_num_frames",
                "last_chunk_fps",
                "peak_vram_allocated_gb",
                "peak_vram_reserved_gb",
                "kv_bank_blocks",
                "kv_bank_evicted_blocks",
                "kv_bank_total_bytes",
                "kv_bank_total_gb",
                "kv_bank_branches",
                "output_path",
            ],
        )
        _writer.writeheader()
    retrieval_fields = [
        "sample_order",
        "prompt_index",
        "branch",
        "current_frame_start",
        "current_num_frames",
        "metric",
        "granularity",
        "retrieval_frames",
        "rope_correction",
        "prope_reencode_mode",
        "prope_reencoded",
        "candidate_block_ids",
        "selected_block_ids",
        "selected_frame_starts",
        "distances",
        "retrieved_tokens_per_layer",
        "selection_seconds",
        "payload_seconds",
    ]
    with open(retrieval_csv_path, "w", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=retrieval_fields).writeheader()
    open(retrieval_jsonl_path, "w", encoding="utf-8").close()

# Load per-prompt trajectory list if provided
trajectory_list = None
if args.trajectory_path:
    with open(args.trajectory_path, encoding="utf-8") as _f:
        trajectory_list = [line.strip() for line in _f if line.strip()]
    assert len(trajectory_list) > selected_indices[-1], (
        f"trajectory_path has {len(trajectory_list)} lines but selected prompt index "
        f"{selected_indices[-1]} requires at least {selected_indices[-1] + 1} lines"
    )

def resolve_camera(idx: int, requested_num_frames: int):
    sample_num_frames = requested_num_frames
    viewmats = None
    Ks = None
    traj_str = None

    if trajectory_list:
        traj_str = trajectory_list[idx]
    elif args.trajectory:
        traj_str = args.trajectory
    if traj_str:
        viewmats, Ks = make_camera_tensors(
            traj_str,
            device=device,
            dtype=torch.bfloat16,
        )
        if requested_num_frames > 0:
            if viewmats.shape[1] < requested_num_frames:
                raise ValueError(
                    f"trajectory has {viewmats.shape[1]} frames, but "
                    f"num_output_frames={requested_num_frames}"
                )
            viewmats = viewmats[:, :requested_num_frames]
            Ks = Ks[:, :requested_num_frames]
        sample_num_frames = int(viewmats.shape[1])

    block_size = int(config.get("num_frame_per_block", 1))
    if block_size > 1 and sample_num_frames % block_size != 0:
        raise ValueError(
            f"num_output_frames={sample_num_frames} must be divisible by "
            f"num_frame_per_block={block_size}"
        )

    return viewmats, Ks, traj_str, sample_num_frames

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


# Latency bookkeeping (rank 0 only; first prompt is recorded as None to skip warmup).
# All pipelines expose `last_chunk0_latency`: time from sampling start to the first
# denoised latent ready, EXCLUDING VAE decode — matches HY15 latency definition.
chunk0_latencies = []


for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    sample_t0 = time.perf_counter()
    idx = batch_data['idx'].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    prompt = batch['prompts'][0]
    viewmats, Ks, traj_str, sample_num_output_frames = resolve_camera(
        idx, args.num_output_frames
    )
    pose_summary = summarize_camera_pose(viewmats)
    if local_rank == 0:
        print(
            f"[camera-pose] sample={i} prompt_index={idx} "
            f"trajectory={traj_str or ''} "
            f"frames={pose_summary['camera_pose_frames']} "
            f"trans_range={pose_summary['camera_translation_range']} "
            f"trans_final={pose_summary['camera_translation_final']} "
            f"rot_range_deg={pose_summary['camera_rotation_range_deg']} "
            f"rot_final_deg={pose_summary['camera_rotation_final_deg']} "
            f"first_motion_frame={pose_summary['camera_first_motion_frame']}",
            flush=True,
        )
    if sample_num_output_frames <= 0:
        raise ValueError("--num_output_frames must be > 0")
    camera_suffix = ""
    if traj_str:
        camera_suffix = "_" + traj_str.replace("*", "").replace(",", "")
    output_path = os.path.join(args.output_folder, safe_video_filename(prompt, camera_suffix))

    if args.i2v:
        assert config.num_frame_per_block == 1, "Current I2V only supports the frame-wise model."
        # For image-to-video, batch contains image and caption
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            if local_rank == 0:
                row = {
                    "sample_order": i,
                    "prompt_index": idx,
                    "status": "skipped_exists",
                    "prompt": prompt,
                    "trajectory": traj_str or "",
                    "num_output_frames": sample_num_output_frames,
                    **pose_summary,
                    "num_generated_latent_frames": 0,
                    "generation_seconds": 0.0,
                    "postprocess_seconds": 0.0,
                    "write_video_seconds": 0.0,
                    "total_seconds": time.perf_counter() - sample_t0,
                    "chunk0_latency_seconds": "",
                    "last_chunk_seconds": "",
                    "last_chunk_num_frames": "",
                    "last_chunk_fps": "",
                    "peak_vram_allocated_gb": "",
                    "peak_vram_reserved_gb": "",
                    "kv_bank_blocks": "",
                    "kv_bank_evicted_blocks": "",
                    "kv_bank_total_bytes": "",
                    "kv_bank_total_gb": "",
                    "kv_bank_branches": "",
                    "output_path": output_path,
                }
                timing_rows.append(row)
                with open(timing_csv_path, "a", newline="", encoding="utf-8") as _f:
                    csv.DictWriter(_f, fieldnames=row.keys()).writerow(row)
            continue
        # Process the image
        image = batch['image'].squeeze(0).unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)

        # Encode the input image as the first latent
        initial_latent = pipeline.vae.encode_to_latent(image).to(device=device, dtype=torch.bfloat16)
        prompts = [prompt] 
        sampled_noise = torch.randn(
            [1, sample_num_output_frames - 1, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
    else:
        # For text-to-video, batch is just the text prompt
        if os.path.exists(output_path):
            print('Video has been generated. Pass!')
            if local_rank == 0:
                row = {
                    "sample_order": i,
                    "prompt_index": idx,
                    "status": "skipped_exists",
                    "prompt": prompt,
                    "trajectory": traj_str or "",
                    "num_output_frames": sample_num_output_frames,
                    **pose_summary,
                    "num_generated_latent_frames": 0,
                    "generation_seconds": 0.0,
                    "postprocess_seconds": 0.0,
                    "write_video_seconds": 0.0,
                    "total_seconds": time.perf_counter() - sample_t0,
                    "chunk0_latency_seconds": "",
                    "last_chunk_seconds": "",
                    "last_chunk_num_frames": "",
                    "last_chunk_fps": "",
                    "peak_vram_allocated_gb": "",
                    "peak_vram_reserved_gb": "",
                    "kv_bank_blocks": "",
                    "kv_bank_evicted_blocks": "",
                    "kv_bank_total_bytes": "",
                    "kv_bank_total_gb": "",
                    "kv_bank_branches": "",
                    "output_path": output_path,
                }
                timing_rows.append(row)
                with open(timing_csv_path, "a", newline="", encoding="utf-8") as _f:
                    csv.DictWriter(_f, fieldnames=row.keys()).writerow(row)
            continue
        extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
        if extended_prompt is not None:
            prompts = [extended_prompt] 
        else:
            prompts = [prompt] 

        initial_latent = None
        sampled_noise = torch.randn(
            [1, sample_num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )

    # Generate frames
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    generation_t0 = time.perf_counter()
    video, latents = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=initial_latent,
        viewmats=viewmats,
        Ks=Ks
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_t0
    peak_vram_allocated_gb = ""
    peak_vram_reserved_gb = ""
    if torch.cuda.is_available():
        peak_vram_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        peak_vram_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    # Record latency on rank 0; first prompt is warmup → None.
    # All pipelines stop the timer before VAE decode (see pipeline.last_chunk0_latency).
    sample_lat = None
    if local_rank == 0:
        sample_lat = getattr(pipeline, "last_chunk0_latency", None)
        if len(chunk0_latencies) >= 1:
            chunk0_latencies.append(sample_lat)
        else:
            chunk0_latencies.append(None)

    postprocess_t0 = time.perf_counter()
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    clean_latent = latents[0].cpu() 
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    postprocess_seconds = time.perf_counter() - postprocess_t0

    write_video_seconds = 0.0
    if not (args.sp_size > 1 and local_rank != 0):
        write_t0 = time.perf_counter()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_video(output_path, video[0], fps=16)
        write_video_seconds = time.perf_counter() - write_t0
    if dist.is_initialized():
        dist.barrier()

    if local_rank == 0:
        bank_summary = pipeline.kv_bank.summary() if hasattr(pipeline, "kv_bank") else {}
        row = {
            "sample_order": i,
            "prompt_index": idx,
            "status": "generated",
            "prompt": prompt,
            "trajectory": traj_str or "",
            "num_output_frames": sample_num_output_frames,
            **pose_summary,
            "num_generated_latent_frames": num_generated_frames,
            "generation_seconds": generation_seconds,
            "postprocess_seconds": postprocess_seconds,
            "write_video_seconds": write_video_seconds,
            "total_seconds": time.perf_counter() - sample_t0,
            "chunk0_latency_seconds": sample_lat if sample_lat is not None and len(chunk0_latencies) > 1 else "",
            "last_chunk_seconds": getattr(pipeline, "last_chunk_seconds", ""),
            "last_chunk_num_frames": getattr(pipeline, "last_chunk_num_frames", ""),
            "last_chunk_fps": getattr(pipeline, "last_chunk_fps", ""),
            "peak_vram_allocated_gb": peak_vram_allocated_gb,
            "peak_vram_reserved_gb": peak_vram_reserved_gb,
            "kv_bank_blocks": bank_summary.get("blocks", 0),
            "kv_bank_evicted_blocks": bank_summary.get("evicted_blocks", 0),
            "kv_bank_total_bytes": bank_summary.get("total_bytes", 0),
            "kv_bank_total_gb": bank_summary.get("total_gb", 0.0),
            "kv_bank_branches": ",".join(bank_summary.get("branches", [])),
            "output_path": output_path,
        }
        timing_rows.append(row)
        with open(timing_csv_path, "a", newline="", encoding="utf-8") as _f:
            csv.DictWriter(_f, fieldnames=row.keys()).writerow(row)
        retrieval_events = getattr(pipeline, "last_retrieval_events", [])
        if retrieval_events:
            with open(retrieval_jsonl_path, "a", encoding="utf-8") as _jsonl, \
                    open(retrieval_csv_path, "a", newline="", encoding="utf-8") as _csv:
                retrieval_fields = [
                    "sample_order",
                    "prompt_index",
                    "branch",
                    "current_frame_start",
                    "current_num_frames",
                    "metric",
                    "granularity",
                    "retrieval_frames",
                    "rope_correction",
                    "prope_reencode_mode",
                    "prope_reencoded",
                    "candidate_block_ids",
                    "selected_block_ids",
                    "selected_frame_starts",
                    "distances",
                    "retrieved_tokens_per_layer",
                    "selection_seconds",
                    "payload_seconds",
                ]
                writer = csv.DictWriter(_csv, fieldnames=retrieval_fields)
                for event in retrieval_events:
                    event_row = {
                        "sample_order": i,
                        "prompt_index": idx,
                        **event,
                    }
                    retrieval_event_rows.append(event_row)
                    _jsonl.write(json.dumps(event_row, ensure_ascii=False) + "\n")
                    csv_row = dict(event_row)
                    for key in ("candidate_block_ids", "selected_block_ids", "selected_frame_starts", "distances"):
                        csv_row[key] = json.dumps(csv_row.get(key, []), ensure_ascii=False)
                    writer.writerow(csv_row)
        print(
            f"[video-timing] sample={i} prompt_index={idx} status=generated "
            f"generation={generation_seconds:.3f}s postprocess={postprocess_seconds:.3f}s "
            f"write_video={write_video_seconds:.3f}s total={row['total_seconds']:.3f}s "
            f"output={output_path}"
        )


# Aggregate latency on rank 0 (drop the first prompt's warmup).
if local_rank == 0:
    valid = [v for v in chunk0_latencies[1:] if v is not None]
    if valid:
        print(f"[timing] rank0 chunk0 latency excl. decode (from 2nd prompt): "
              f"avg={sum(valid)/len(valid):.3f}s over {len(valid)} samples")
    with open(timing_json_path, "w", encoding="utf-8") as _f:
        json.dump(timing_rows, _f, indent=2, ensure_ascii=False)

       
