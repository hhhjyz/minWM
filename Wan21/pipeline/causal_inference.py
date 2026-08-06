import json
import time
from typing import List, Optional
import torch

from wan_utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper

from demo_utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation
import tqdm
from .debug_utils import log_cache_state, should_log_cache
from .kv_bank import KVBank, normalize_retrieval_granularity
from .sink_utils import get_model_sink_size_frames, is_bank_sink_strategy, maybe_update_sink, normalize_sink_strategy

class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.prope_kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size
        self.log_cache_state = bool(getattr(args, "log_cache_state", False))
        self.log_cache_interval = int(getattr(args, "log_cache_interval", 1))
        self.sink_strategy = normalize_sink_strategy(getattr(args, "sink_strategy", "none"))
        self.sink_update_interval = int(getattr(args, "sink_update_interval", 0))
        self.sink_bank_seed = int(getattr(args, "sink_bank_seed", 0))
        self.sink_size_frames = get_model_sink_size_frames(self.generator)
        self.retrieval_enable = bool(getattr(args, "retrieval_enable", False))
        self.tri_region_rope_rebase = bool(
            getattr(self.generator.model, "tri_region_rope_rebase", False)
        )
        self.rope_train_length = int(
            getattr(self.generator.model, "rope_train_length", 19)
        )
        self.retrieval_granularity = normalize_retrieval_granularity(
            getattr(args, "retrieval_granularity", "chunk")
        )
        self.retrieval_metric = str(getattr(args, "retrieval_metric", "pose"))
        self.retrieval_frames = max(0, int(getattr(args, "retrieval_frames", 0)))
        self.retrieval_recent_frames = max(0, int(getattr(args, "retrieval_recent_frames", 0)))
        self.retrieval_fov_samples = max(1, int(getattr(args, "retrieval_fov_samples", 8192)))
        self.retrieval_fov_radius = float(getattr(args, "retrieval_fov_radius", 8.0))
        self.retrieval_fov_h_deg = float(getattr(args, "retrieval_fov_h_deg", 60.0))
        self.retrieval_fov_v_deg = float(getattr(args, "retrieval_fov_v_deg", 35.0))
        self.retrieval_hybrid_fov_weight = float(getattr(args, "retrieval_hybrid_fov_weight", 0.5))
        self.retrieval_rope_correction = bool(getattr(args, "retrieval_rope_correction", False))
        self.prope_reencode_mode = str(getattr(args, "prope_reencode_mode", "none"))
        self.kv_compression_enable = bool(getattr(args, "kv_compression_enable", False))
        self.kv_compression_keep_ratio = float(getattr(args, "kv_compression_keep_ratio", 0.5))
        self.kv_compression_at_store = bool(getattr(args, "kv_compression_at_store", False))
        self.kv_compression_anchor_rotate = bool(getattr(args, "kv_compression_anchor_rotate", False))
        self.kv_compression_pooled = bool(getattr(args, "kv_compression_pooled", False))
        self.kv_compression_dynamic_enable = bool(getattr(args, "kv_compression_dynamic_enable", False))
        if self.retrieval_granularity == "latent_frame" and self.kv_compression_at_store:
            raise ValueError(
                "latent_frame retrieval requires uncompressed bank entries; "
                "disable kv_compression_at_store"
            )
        self.kv_bank = KVBank(
            enabled=bool(getattr(args, "kv_bank_enable", False)) or self.retrieval_enable or is_bank_sink_strategy(self.sink_strategy),
            device=str(getattr(args, "kv_bank_device", "cpu")),
            max_blocks=int(getattr(args, "kv_bank_max_blocks", 0)),
            log_interval=int(getattr(args, "kv_bank_log_interval", 1)),
            warn_memory_gb=float(getattr(args, "kv_bank_warn_memory_gb", 16.0)),
            compression_enable=self.kv_compression_enable,
            compression_keep_ratio=self.kv_compression_keep_ratio,
            compression_at_store=self.kv_compression_at_store,
            compression_pooled=self.kv_compression_pooled,
            compression_dynamic_enable=self.kv_compression_dynamic_enable,
            compression_dynamic_min_keep=float(getattr(args, "kv_compression_dynamic_min_keep", 0.25)),
            compression_dynamic_max_keep=float(getattr(args, "kv_compression_dynamic_max_keep", 0.75)),
            compression_dynamic_translation_scale=float(getattr(args, "kv_compression_dynamic_translation_scale", 1.0)),
            compression_dynamic_rotation_scale=float(getattr(args, "kv_compression_dynamic_rotation_scale", 0.35)),
            compression_dynamic_motion_weight=float(getattr(args, "kv_compression_dynamic_motion_weight", 0.25)),
        )
        if self.sink_strategy != "none":
            print(
                f"Sink strategy: {self.sink_strategy} "
                f"(sink_size={self.sink_size_frames}, update_interval={self.sink_update_interval})"
            )
        if self.kv_bank.enabled:
            print(
                f"KV bank enabled: device={self.kv_bank.device_name} "
                f"max_blocks={self.kv_bank.max_blocks or 'unlimited'}"
            )
        if self.retrieval_enable:
            print(
                f"Retrieval window enabled: metric={self.retrieval_metric} "
                f"granularity={self.retrieval_granularity} "
                f"retrieval_frames={self.retrieval_frames} recent_exclusion={self.retrieval_recent_frames}"
            )
            if self.tri_region_rope_rebase:
                print(
                    f"Tri-region warm-up: retrieval/rebase start at "
                    f"current_start_frame >= {self.rope_train_length}"
                )

        # Latency of producing the first chunk (set by inference()).
        self.last_chunk0_latency = None
        self.last_chunk_seconds = None
        self.last_chunk_num_frames = None
        self.last_chunk_fps = None
        self.last_retrieval_events = []

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def _record_retrieval_event(self, event):
        self.last_retrieval_events.append(event)
        print(f"[retrieval-event] {json.dumps(event, ensure_ascii=False)}", flush=True)

    def _build_retrieval_payloads(
        self,
        *,
        branch: str,
        current_start_frame: int,
        current_num_frames: int,
        viewmats,
        Ks,
        device,
    ):
        if not self.retrieval_enable or self.retrieval_frames <= 0:
            return None
        # Keep retrieval disabled during the ordinary-RoPE warm-up.  The
        # model enables tri-region rebase at this same T boundary.
        if (
            self.tri_region_rope_rebase
            and current_start_frame < self.rope_train_length
        ):
            return None
        retrieval_t0 = time.perf_counter()
        selector = (
            self.kv_bank.select_retrieval_frames
            if self.retrieval_granularity == "latent_frame"
            else self.kv_bank.select_retrieval_blocks
        )
        selection_kwargs = dict(
            current_viewmats=viewmats,
            current_frame_start=current_start_frame,
            retrieval_frames=self.retrieval_frames,
            metric=self.retrieval_metric,
            fov_samples=self.retrieval_fov_samples,
            fov_radius=self.retrieval_fov_radius,
            fov_h_deg=self.retrieval_fov_h_deg,
            fov_v_deg=self.retrieval_fov_v_deg,
            hybrid_fov_weight=self.retrieval_hybrid_fov_weight,
            recent_frames=self.retrieval_recent_frames,
            sink_size_frames=self.sink_size_frames,
            return_details=True,
        )
        if self.retrieval_granularity == "chunk":
            selection_kwargs["current_frame_count"] = current_num_frames
            selection_kwargs["device"] = device
        selected, details = selector(**selection_kwargs)
        selection_seconds = time.perf_counter() - retrieval_t0
        if not selected:
            self._record_retrieval_event({
                "branch": branch,
                "current_frame_start": current_start_frame,
                "current_num_frames": current_num_frames,
                "metric": self.retrieval_metric,
                "granularity": self.retrieval_granularity,
                "retrieval_frames": self.retrieval_frames,
                "rope_correction": self.retrieval_rope_correction,
                "prope_reencode_mode": self.prope_reencode_mode,
                "prope_reencoded": False,
                "candidate_block_ids": details.get("candidate_block_ids", []),
                "selected_block_ids": [],
                "selected_frame_starts": [],
                "distances": details.get("distances", []),
                "retrieved_tokens_per_layer": 0,
                "selection_seconds": selection_seconds,
                "payload_seconds": 0.0,
            })
            return None
        payload_t0 = time.perf_counter()
        payload_kwargs = dict(
            branch=branch, device=device, include_prope=viewmats is not None,
            frame_seq_length=self.frame_seq_length,
            rope_correction=self.retrieval_rope_correction,
            prope_reencode_mode=self.prope_reencode_mode,
            current_viewmats=viewmats, current_Ks=Ks,
        )
        if self.retrieval_granularity == "latent_frame":
            payloads = self.kv_bank.get_frame_retrieval_payloads(selected, **payload_kwargs)
        else:
            payloads = self.kv_bank.get_retrieval_payloads(
                selected,
                compress_runtime=self.kv_compression_enable and not self.kv_compression_at_store,
                compression_keep_ratio=self.kv_compression_keep_ratio,
                compression_anchor_rotate=self.kv_compression_anchor_rotate,
                compression_pooled=self.kv_compression_pooled,
                **payload_kwargs,
            )
        payload_seconds = time.perf_counter() - payload_t0
        retrieved_tokens = int(payloads[0]["k"].shape[1]) if payloads else 0
        self._record_retrieval_event({
            "branch": branch,
            "current_frame_start": current_start_frame,
            "current_num_frames": current_num_frames,
            "metric": self.retrieval_metric,
            "granularity": self.retrieval_granularity,
            "retrieval_frames": self.retrieval_frames,
            "rope_correction": self.retrieval_rope_correction,
            "prope_reencode_mode": self.prope_reencode_mode,
            "prope_reencoded": bool(payloads[0].get("prope_reencoded", False)) if payloads else False,
            "candidate_block_ids": details.get("candidate_block_ids", []),
            "selected_block_ids": details.get("selected_block_ids", []),
            "selected_frame_starts": details.get("selected_frame_starts", []),
            "distances": details.get("distances", []),
            "retrieved_tokens_per_layer": retrieved_tokens,
            "selection_seconds": selection_seconds,
            "payload_seconds": payload_seconds,
        })
        return payloads

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        rectified_tf = False,
        viewmats: Optional[torch.Tensor] = None,  # (B, F, 4, 4) PRoPE camera extrinsics
        Ks: Optional[torch.Tensor] = None,         # (B, F, 3, 3) PRoPE camera intrinsics
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # default here
            # self.independent_first_frame: False
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Start chunk0 latency timer AFTER text encoder, BEFORE VAE decode
        # — matches HY15 latency definition (excludes both text encoder and decode).
        torch.cuda.synchronize()
        _chunk0_t0 = time.perf_counter()
        self.last_chunk0_latency = None
        self.last_chunk_seconds = None
        self.last_chunk_num_frames = None
        self.last_chunk_fps = None
        self.last_retrieval_events = []

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_prope_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            # reset prope kv cache
            for block_index in range(len(self.prope_kv_cache1)):
                self.prope_kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.prope_kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                vm_chunk = viewmats[:, 0:1] if viewmats is not None else None
                ks_chunk = Ks[:, 0:1] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache1,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                vm_chunk = viewmats[:, current_start_frame:current_start_frame + self.num_frame_per_block] if viewmats is not None else None
                ks_chunk = Ks[:, current_start_frame:current_start_frame + self.num_frame_per_block] if Ks is not None else None
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    viewmats=vm_chunk,
                    Ks=ks_chunk,
                    prope_kv_cache=self.prope_kv_cache1,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        self.kv_bank.reset(expected_blocks=len(all_num_frames))
        for block_index, current_num_frames in enumerate(tqdm.tqdm(all_num_frames)):
            torch.cuda.synchronize()
            _block_t0 = time.perf_counter()
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            vm_chunk = viewmats[:, current_start_frame:current_start_frame + current_num_frames] if viewmats is not None else None
            ks_chunk = Ks[:, current_start_frame:current_start_frame + current_num_frames] if Ks is not None else None
            retrieval_kv = self._build_retrieval_payloads(
                branch="main",
                current_start_frame=current_start_frame,
                current_num_frames=current_num_frames,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                device=noise.device,
            )
            if should_log_cache(self.log_cache_state, block_index, self.log_cache_interval):
                log_cache_state(
                    tag="before_denoise",
                    block_index=block_index,
                    current_start_frame=current_start_frame,
                    current_num_frames=current_num_frames,
                    frame_seq_length=self.frame_seq_length,
                    kv_cache=self.kv_cache1,
                    prope_kv_cache=self.prope_kv_cache1,
                    viewmats=vm_chunk,
                    device=noise.device,
                )

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")
                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        prope_kv_cache=self.prope_kv_cache1,
                        retrieval_kv=retrieval_kv,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        viewmats=vm_chunk,
                        Ks=ks_chunk,
                        prope_kv_cache=self.prope_kv_cache1,
                        retrieval_kv=retrieval_kv,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Capture chunk0 latency: stop timer once the first chunk's denoised output is ready.
            if self.last_chunk0_latency is None:
                torch.cuda.synchronize()
                self.last_chunk0_latency = time.perf_counter() - _chunk0_t0

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                viewmats=vm_chunk,
                Ks=ks_chunk,
                prope_kv_cache=self.prope_kv_cache1,
                retrieval_kv=retrieval_kv,
            )
            self.kv_bank.append_block(
                block_id=block_index,
                frame_start=current_start_frame,
                frame_count=current_num_frames,
                frame_seq_length=self.frame_seq_length,
                cache_branches={
                    "main": (
                        self.kv_cache1,
                        self.prope_kv_cache1 if viewmats is not None else None,
                    )
                },
                viewmats=vm_chunk,
                Ks=ks_chunk,
            )
            maybe_update_sink(
                strategy=self.sink_strategy,
                update_interval=self.sink_update_interval,
                block_index=block_index,
                current_frame_start=current_start_frame,
                current_num_frames=current_num_frames,
                frame_seq_length=self.frame_seq_length,
                sink_size_frames=self.sink_size_frames,
                kv_cache=self.kv_cache1,
                prope_kv_cache=self.prope_kv_cache1,
                kv_bank=self.kv_bank,
                branch="main",
                current_viewmats=vm_chunk,
                current_Ks=ks_chunk,
                recent_frames=self.retrieval_recent_frames,
                fov_samples=self.retrieval_fov_samples,
                fov_radius=self.retrieval_fov_radius,
                fov_h_deg=self.retrieval_fov_h_deg,
                fov_v_deg=self.retrieval_fov_v_deg,
                hybrid_fov_weight=self.retrieval_hybrid_fov_weight,
                random_seed=self.sink_bank_seed,
                prope_reencode_mode=self.prope_reencode_mode,
                label="causal",
            )
            if should_log_cache(self.log_cache_state, block_index, self.log_cache_interval):
                log_cache_state(
                    tag="after_clean_update",
                    block_index=block_index,
                    current_start_frame=current_start_frame,
                    current_num_frames=current_num_frames,
                    frame_seq_length=self.frame_seq_length,
                    kv_cache=self.kv_cache1,
                    prope_kv_cache=self.prope_kv_cache1,
                    viewmats=vm_chunk,
                    device=noise.device,
                )

            torch.cuda.synchronize()
            block_seconds = time.perf_counter() - _block_t0
            self.last_chunk_seconds = block_seconds
            self.last_chunk_num_frames = current_num_frames
            self.last_chunk_fps = current_num_frames / block_seconds if block_seconds > 0 else None

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()
        if rectified_tf: 
            mean = torch.load('laboratory/mean.pt').to(output.device) 
            std = torch.load('laboratory/std.pt').to(output.device) 
            noise = torch.randn_like(output).to(output.device) 
            output -= mean 
        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output, use_cache=False)
        # A 25 s MBench-A sample decodes to a multi-GiB tensor.  The
        # out-of-place expression previously used here allocated another
        # tensor of the same size and exhausted 48 GiB GPUs after decoding.
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        Under SP, KV cache is stored in head-parallel domain (post all-to-all),
        so each rank only stores num_heads // sp_size heads.
        """
        num_heads = self._get_sp_num_heads(12)
        kv_cache1 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = 31200

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.

        NOTE: Cross-attention does NOT use SP all-to-all, so cache keeps full num_heads.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _initialize_prope_kv_cache(self, batch_size, dtype, device):
        num_heads = self._get_sp_num_heads(12)
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 31200
        self.prope_kv_cache1 = [{
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        } for _ in range(self.num_transformer_blocks)]

    @staticmethod
    def _get_sp_num_heads(full_num_heads):
        """Return per-rank num_heads under SP (head-parallel domain)."""
        try:
            from sp.parallel_states import get_parallel_state
            ps = get_parallel_state()
            if ps.sp_enabled:
                return full_num_heads // ps.sp
        except (ImportError, AttributeError):
            pass
        return full_num_heads
