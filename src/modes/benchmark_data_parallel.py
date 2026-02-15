"""Data-parallel benchmark: each GPU independently runs all diffusion steps.

Provides a baseline comparison against the pipeline-parallel benchmark
(src/modes/benchmark.py). In data-parallel mode every GPU holds the full
model and processes a disjoint subset of samples with no inter-GPU
communication during inference.

Output format matches benchmark.py (BENCHMARK_JSON=...) for easy comparison.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import torch
import torch.distributed as dist

from ..distributed.backend import resolve_backend
from ..distributed.setup import finalize_distributed, init_distributed

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data parallel throughput benchmark")
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


def _run_all_steps(model, latent: torch.Tensor, total_steps: int) -> torch.Tensor:
    """Run all diffusion steps sequentially on a single GPU."""
    for step in range(total_steps):
        latent = model(latent, step)
    return latent


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

    latent_shape = torch.Size(
        (1, args.latent_channels, args.latent_frames, args.latent_height, args.latent_width)
    )

    init_noise_sigma = model.init_noise_sigma if use_svd else 1.0

    # Distribute measured samples across ranks
    if args.num_samples % world_size != 0:
        measured_per_rank = (args.num_samples + world_size - 1) // world_size
    else:
        measured_per_rank = args.num_samples // world_size

    LOGGER.info(
        "Rank %d: %d warmup + %d measured samples, %d total steps each",
        rank,
        args.warmup_samples,
        measured_per_rank,
        args.total_steps,
    )

    # --- Phase 1: Warmup (each rank runs warmup_samples locally) ---
    LOGGER.info("Rank %d: running %d warmup samples...", rank, args.warmup_samples)
    torch.cuda.synchronize(device)
    dist.barrier()

    with torch.no_grad():
        for wi in range(args.warmup_samples):
            torch.manual_seed(args.seed + rank * 10000 + wi)
            latent = torch.randn(latent_shape, device=device, dtype=dtype) * init_noise_sigma
            _run_all_steps(model, latent, args.total_steps)

    torch.cuda.synchronize(device)
    dist.barrier()
    LOGGER.info("Rank %d: warmup complete", rank)

    # --- Phase 2: Measured (each rank processes its share) ---
    my_start = rank * measured_per_rank
    my_end = min(my_start + measured_per_rank, args.num_samples)
    my_count = my_end - my_start

    per_sample_times: list[float] = []
    measured_start = time.perf_counter()

    with torch.no_grad():
        for sample_idx in range(my_start, my_end):
            sample_start = time.perf_counter()

            torch.manual_seed(args.seed + sample_idx)
            latent = torch.randn(latent_shape, device=device, dtype=dtype) * init_noise_sigma
            _run_all_steps(model, latent, args.total_steps)

            torch.cuda.synchronize(device)
            per_sample_times.append(time.perf_counter() - sample_start)

    measured_elapsed = time.perf_counter() - measured_start

    # Synchronize after measurement
    dist.barrier()

    # Gather timing from all ranks to rank 0 via send/recv
    timing_tensor = torch.tensor(
        [measured_elapsed, float(my_count)], device=device, dtype=torch.float64
    )

    if rank == 0:
        all_timings: list[torch.Tensor] = []
        all_timings.append(timing_tensor.clone())
        for src in range(1, world_size):
            buf = torch.zeros(2, device=device, dtype=torch.float64)
            dist.recv(buf, src=src)
            all_timings.append(buf)
    else:
        dist.send(timing_tensor, dst=0)

    # Rank 0 reports results
    if rank == 0:
        # Wall-clock time is the max across all ranks (all start together after barrier)
        max_elapsed = max(t[0].item() for t in all_timings)
        total_measured = sum(int(t[1].item()) for t in all_timings)

        avg_sample_time = sum(per_sample_times) / len(per_sample_times) if per_sample_times else 0.0
        throughput = total_measured / max_elapsed if max_elapsed > 0 else 0.0
        first_sample_time = per_sample_times[0] if per_sample_times else 0.0

        results = {
            "mode": "data_parallel",
            "world_size": world_size,
            "total_steps": args.total_steps,
            "steps_per_gpu": args.total_steps,
            "model": args.model,
            "num_samples_measured": total_measured,
            "warmup_samples": args.warmup_samples,
            "samples_per_rank": measured_per_rank,
            "latent_shape": list(latent_shape),
            "first_sample_time_s": round(first_sample_time, 4),
            "avg_sample_time_s": round(avg_sample_time, 4),
            "throughput_samples_per_s": round(throughput, 4),
            "wall_clock_s": round(max_elapsed, 4),
            "per_sample_times_ms": [round(t * 1000, 2) for t in per_sample_times],
        }

        LOGGER.info("=" * 70)
        LOGGER.info("DATA PARALLEL BENCHMARK RESULTS")
        LOGGER.info("=" * 70)
        LOGGER.info(
            "GPUs: %d | Steps/GPU: %d (all) | Model: %s | Samples: %d (+ %d warmup/rank)",
            world_size,
            args.total_steps,
            args.model,
            total_measured,
            args.warmup_samples,
        )
        LOGGER.info("Latent: %s (%s)", list(latent_shape), dtype)
        LOGGER.info("Samples per rank: %d", measured_per_rank)
        LOGGER.info("-" * 70)
        LOGGER.info("First sample (rank 0): %.2f s", first_sample_time)
        LOGGER.info("Avg sample (rank 0):   %.4f s", avg_sample_time)
        LOGGER.info("Wall clock (measured): %.4f s", max_elapsed)
        LOGGER.info("Throughput:            %.4f samples/s", throughput)
        LOGGER.info("-" * 70)
        LOGGER.info(
            "Per-sample rank 0 (ms): %s",
            [round(t * 1000, 1) for t in per_sample_times],
        )
        LOGGER.info("=" * 70)

        print(f"BENCHMARK_JSON={json.dumps(results)}")

    finalize_distributed()


if __name__ == "__main__":
    main()
