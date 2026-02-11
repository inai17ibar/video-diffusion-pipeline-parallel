"""Production mode entrypoint for multi-GPU NCCL execution."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

import torch

from ..distributed.backend import resolve_backend
from ..distributed.setup import finalize_distributed, init_distributed
from ..models.svd_unet import StableVideoUNet
from ..pipeline.pipeline import LatentSpec, run_pipeline_latents

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production pipeline mode")
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--latent-shape", type=int, nargs=5, metavar=("B", "C", "F", "H", "W"))
    parser.add_argument("--timesteps", type=int, nargs="+", help="Explicit timestep schedule")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--backend", type=str, default="auto")
    parser.add_argument("--init-method", type=str, default=None)
    parser.add_argument(
        "--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt"
    )
    parser.add_argument("--fps", type=int, default=6, help="Frames per second for conditioning")
    parser.add_argument(
        "--motion-bucket-id", type=int, default=127, help="Motion bucket ID (0-255)"
    )
    parser.add_argument(
        "--enable-memory-opt",
        action="store_true",
        help="Enable memory optimizations (xformers/attention slicing)",
    )
    parser.add_argument(
        "--attention-slicing",
        action="store_true",
        help="Enable attention slicing for reduced memory",
    )
    return parser.parse_args()


def _discover_rank() -> int:
    return int(os.environ["RANK"])


def _discover_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def _discover_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


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

    # Build timestep schedule
    if args.timesteps:
        timesteps: Sequence[int] = args.timesteps
    else:
        # Generate default descending schedule
        timesteps = list(range(args.total_steps - 1, -1, -1))

    LOGGER.info("Loading model from %s on rank %d", args.model_id, rank)
    model = StableVideoUNet.from_pretrained(
        model_id=args.model_id,
        timesteps=timesteps,
        torch_dtype=torch.float16,
        enable_memory_efficient_attention=args.enable_memory_opt,
        enable_sliced_attention=args.attention_slicing,
    ).to(device)

    # Apply additional memory optimizations if requested
    if args.enable_memory_opt:
        model.enable_memory_optimizations()
        LOGGER.info("Memory optimizations enabled on rank %d", rank)

    # Set dummy conditioning for benchmarking
    # latent_shape is (B, C, F, H, W) where C should be 4
    batch_size = args.latent_shape[0] if args.latent_shape else 1
    num_frames = args.latent_shape[2] if args.latent_shape else 14
    height = args.latent_shape[3] if args.latent_shape else 64
    width = args.latent_shape[4] if args.latent_shape else 64
    model.set_dummy_conditioning(
        batch_size=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
    )
    LOGGER.info("Model loaded and conditioning set on rank %d", rank)

    latent_spec = LatentSpec(
        shape=torch.Size(tuple(args.latent_shape)), dtype=torch.float16, device=device
    )

    init_noise_sigma = model.init_noise_sigma

    def _input_supplier(sample_idx: int) -> torch.Tensor:
        torch.manual_seed(args.seed + sample_idx)
        return (
            torch.randn(latent_spec.shape, device=device, dtype=latent_spec.dtype)
            * init_noise_sigma
        )

    run_pipeline_latents(
        model,
        total_steps=args.total_steps,
        timesteps=timesteps,
        world_size=world_size,
        rank=rank,
        latent_spec=latent_spec,
        num_samples=args.num_samples,
        input_supplier=_input_supplier,
    )

    finalize_distributed()


if __name__ == "__main__":
    main()
