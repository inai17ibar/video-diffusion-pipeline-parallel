"""Benchmark mode: multi-GPU throughput measurement.

Supports both DummyUNet (lightweight) and real SVD UNet for measuring
pipeline-parallel throughput scaling across different GPU counts.

Includes an optional FSDP mode (--fsdp) that shards the UNet across GPUs
for memory-limited scenarios where full-model-per-GPU is infeasible.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import time

import torch
import torch.distributed as dist

from ..distributed.backend import resolve_backend
from ..distributed.setup import finalize_distributed, init_distributed
from ..pipeline.pipeline import LatentSpec, PipelineConfig, PipelineStage

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline parallel throughput benchmark")
    parser.add_argument("--total-steps", type=int, default=28)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-frames", type=int, default=14)
    parser.add_argument("--latent-height", type=int, default=40)
    parser.add_argument("--latent-width", type=int, default=72)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--model",
        type=str,
        default="dummy",
        choices=["dummy", "svd"],
        help="Model to benchmark: dummy (DummyUNet) or svd (real SVD UNet)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "gloo", "nccl"],
    )
    parser.add_argument("--init-method", type=str, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--fsdp", action="store_true", help="Use FSDP model sharding")
    return parser.parse_args()


def _discover_rank() -> int:
    return int(os.environ["RANK"])


def _discover_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def _discover_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def _build_dummy_model(args, device):
    from ..models.dummy_unet import DummyUNet

    return DummyUNet(
        channels=args.latent_channels,
        hidden_channels=args.hidden_channels,
    ).to(device)


def _build_svd_model(args, device, total_steps):
    from ..models.svd_unet import StableVideoUNet

    timesteps = StableVideoUNet._default_timestep_schedule(total_steps)
    model = StableVideoUNet.from_pretrained(
        model_id=args.model_id,
        timesteps=timesteps,
        torch_dtype=torch.float16,
        enable_memory_efficient_attention=True,
        enable_sliced_attention=True,
        attention_slice_size="auto",
    ).to(device)
    model.enable_memory_optimizations()

    model.set_dummy_conditioning(
        batch_size=1,
        num_frames=args.latent_frames,
        height=args.latent_height,
        width=args.latent_width,
        device=device,
        guidance_scale=args.guidance_scale,
    )
    return model


def _wrap_unet_with_fsdp(model, local_rank):
    """Wrap model.unet with FSDP for memory-efficient sharded inference."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
    )
    from torch.distributed.fsdp import (
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1_000_000)
    model.unet = FSDP(
        model.unet,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        ),
        device_id=local_rank,
        use_orig_params=True,
    )
    return model


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rank = _discover_rank()
    local_rank = _discover_local_rank()
    world_size = _discover_world_size()

    backend = resolve_backend(None if args.backend == "auto" else args.backend, simulator=False)
    device = torch.device(f"cuda:{local_rank}")

    init_distributed(
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=args.init_method,
    )

    # Build model
    use_svd = args.model == "svd"
    dtype = torch.float16 if use_svd else torch.float32

    if use_svd:
        LOGGER.info("Loading SVD UNet on rank %d...", rank)
        model = _build_svd_model(args, device, args.total_steps)
        LOGGER.info("SVD UNet loaded on rank %d", rank)
    else:
        model = _build_dummy_model(args, device)

    # FSDP wrapping (must happen after model creation, before inference)
    if args.fsdp:
        if not use_svd:
            raise ValueError("FSDP mode is only supported with --model svd")
        LOGGER.info("[rank=%d] Wrapping UNet with FSDP...", rank)
        model = _wrap_unet_with_fsdp(model, local_rank)
        LOGGER.info("[rank=%d] FSDP wrapping complete", rank)

    timesteps = list(range(args.total_steps - 1, -1, -1))

    latent_shape = torch.Size(
        (1, args.latent_channels, args.latent_frames, args.latent_height, args.latent_width)
    )

    total_samples = args.warmup_samples + args.num_samples
    init_noise_sigma = model.init_noise_sigma if use_svd else 1.0

    # Reset peak memory stats before the timed loop
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    # Synchronize before timing
    torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()

    sample_end_times: list[float] = []
    overall_start = time.perf_counter()

    if args.fsdp:
        # FSDP mode: all GPUs execute all steps together (FSDP synchronizes via all-gather)
        LOGGER.info(
            "[rank=%d] Running FSDP mode: all GPUs execute all %d steps", rank, args.total_steps
        )
        with torch.no_grad():
            for sample_idx in range(total_samples):
                torch.manual_seed(args.seed + sample_idx)
                latent = torch.randn(latent_shape, device=device, dtype=dtype) * init_noise_sigma
                for step in range(args.total_steps):
                    latent = model(latent, step)
                torch.cuda.synchronize(device)
                sample_end_times.append(time.perf_counter())
    else:
        # Pipeline-parallel mode (existing behavior)
        latent_spec = LatentSpec(shape=latent_shape, dtype=dtype, device=device)
        config = PipelineConfig(
            total_steps=args.total_steps,
            world_size=world_size,
            rank=rank,
            timesteps=timesteps,
            latent_spec=latent_spec,
        )
        stage = PipelineStage(model=model, config=config)

        def _input_supplier(sample_idx: int) -> torch.Tensor:
            torch.manual_seed(args.seed + sample_idx)
            return torch.randn(latent_shape, device=device, dtype=dtype) * init_noise_sigma

        with torch.no_grad():
            for sample_idx in range(total_samples):
                input_latent = _input_supplier(sample_idx) if rank == 0 else None
                stage._process_single_latent(input_latent, sample_idx=sample_idx)

                if rank == world_size - 1:
                    torch.cuda.synchronize(device)
                    sample_end_times.append(time.perf_counter())

    torch.cuda.synchronize(device)

    # Collect peak memory stats from all ranks
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    LOGGER.info("[rank=%d] Peak GPU memory: %.2f GB", rank, peak_mem_bytes / 1e9)

    peak_tensor = torch.tensor([peak_mem_bytes], dtype=torch.int64, device=device)
    if world_size > 1:
        gathered = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
        dist.all_gather(gathered, peak_tensor)
        all_peak_gb = [float(t.item()) / 1e9 for t in gathered]
    else:
        all_peak_gb = [peak_mem_bytes / 1e9]

    # In FSDP mode, all ranks have timing data; in pipeline mode, only the final rank does
    reporting_rank = 0 if args.fsdp else world_size - 1

    if rank == reporting_rank:
        # Compute per-sample times
        per_sample_times = []
        for i, end_t in enumerate(sample_end_times):
            start_t = overall_start if i == 0 else sample_end_times[i - 1]
            per_sample_times.append(end_t - start_t)

        # Separate warmup and measured samples
        measured_times = per_sample_times[args.warmup_samples :]
        first_sample_time = per_sample_times[0] if per_sample_times else 0.0

        measured_total = sum(measured_times)
        avg_sample_time = measured_total / len(measured_times) if measured_times else 0.0
        throughput = len(measured_times) / measured_total if measured_total > 0 else 0.0

        results = {
            "world_size": world_size,
            "total_steps": args.total_steps,
            "steps_per_gpu": args.total_steps if args.fsdp else args.total_steps // world_size,
            "model": args.model,
            "fsdp": args.fsdp,
            "num_samples_measured": args.num_samples,
            "warmup_samples": args.warmup_samples,
            "latent_shape": list(latent_shape),
            "first_sample_time_s": round(first_sample_time, 4),
            "avg_sample_time_s": round(avg_sample_time, 4),
            "throughput_samples_per_s": round(throughput, 4),
            "per_sample_times_ms": [round(t * 1000, 2) for t in per_sample_times],
            "peak_memory_gb_per_rank": [round(m, 3) for m in all_peak_gb],
            "max_peak_memory_gb": round(max(all_peak_gb), 3),
        }

        mode_str = "FSDP" if args.fsdp else "Pipeline"
        LOGGER.info("=" * 70)
        LOGGER.info("BENCHMARK RESULTS (%s mode)", mode_str)
        LOGGER.info("=" * 70)
        LOGGER.info(
            "GPUs: %d | Steps/GPU: %s | Model: %s | Samples: %d (+ %d warmup)",
            world_size,
            "all" if args.fsdp else str(args.total_steps // world_size),
            args.model,
            args.num_samples,
            args.warmup_samples,
        )
        LOGGER.info("Latent: %s (%s)", list(latent_shape), dtype)
        LOGGER.info("-" * 70)
        LOGGER.info("First sample (fill):   %.2f s", first_sample_time)
        LOGGER.info("Avg sample (steady):   %.4f s", avg_sample_time)
        LOGGER.info("Throughput:            %.4f samples/s", throughput)
        LOGGER.info("-" * 70)
        LOGGER.info(
            "Per-sample (ms): %s",
            [round(t * 1000, 1) for t in per_sample_times],
        )
        LOGGER.info("-" * 70)
        LOGGER.info("Peak memory per rank (GB): %s", results["peak_memory_gb_per_rank"])
        LOGGER.info("Max peak memory (GB):      %.3f", results["max_peak_memory_gb"])
        LOGGER.info("=" * 70)

        print(f"BENCHMARK_JSON={json.dumps(results)}")

    finalize_distributed()


if __name__ == "__main__":
    main()
