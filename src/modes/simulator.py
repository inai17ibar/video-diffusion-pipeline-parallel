"""Simulator mode entrypoint for torchrun on CPU / single GPU."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Sequence

import torch

from ..distributed.backend import resolve_backend
from ..distributed.setup import finalize_distributed, init_distributed
from ..models.dummy_unet import DummyUNet
from ..pipeline.pipeline import LatentSpec, run_single_latent

LOGGER = logging.getLogger(__name__)


def _str_to_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapping[normalized]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline simulator mode")
    parser.add_argument("--total-steps", type=int, default=28)
    parser.add_argument("--rank", type=int, default=0, help="Rank fallback when env vars missing")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--latent-batch", type=int, default=1)
    parser.add_argument("--latent-channels", type=int, default=8)
    parser.add_argument("--latent-frames", type=int, default=8)
    parser.add_argument("--latent-height", type=int, default=32)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device string understood by torch.device (cpu, cuda:0, mps, ...)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "gloo", "nccl"],
        help="Override backend selection",
    )
    parser.add_argument("--init-method", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _discover_rank(default_rank: int) -> int:
    return int(os.environ.get("RANK", default_rank))


def _discover_local_rank(default_local_rank: int = 0) -> int:
    return int(os.environ.get("LOCAL_RANK", default_local_rank))


def _discover_world_size(default_world_size: int) -> int:
    return int(os.environ.get("WORLD_SIZE", default_world_size))


def _build_timesteps(total_steps: int) -> Sequence[int]:
    # Basic descending schedule (T-1 -> 0) mirrors diffusion usage.
    return list(reversed(range(total_steps)))


def _build_initial_latent(args, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    shape = (
        args.latent_batch,
        args.latent_channels,
        args.latent_frames,
        args.latent_height,
        args.latent_width,
    )
    torch.manual_seed(args.seed)
    latent = torch.randn(shape, device=device, dtype=dtype)
    return latent


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rank = _discover_rank(args.rank)
    world_size = _discover_world_size(args.world_size)

    backend_pref = None if args.backend == "auto" else args.backend
    backend = resolve_backend(backend_pref, simulator=True)

    init_distributed(
        backend=backend, rank=rank, world_size=world_size, init_method=args.init_method
    )

    dtype = _str_to_dtype(args.dtype)

    # Handle device assignment for multi-GPU
    local_rank = _discover_local_rank()
    if args.device == "cuda":
        # Use local_rank to assign each process to a different GPU
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)

    model = DummyUNet(channels=args.latent_channels).to(device)
    timesteps = _build_timesteps(args.total_steps)
    latent_spec = LatentSpec(
        shape=torch.Size(
            (
                args.latent_batch,
                args.latent_channels,
                args.latent_frames,
                args.latent_height,
                args.latent_width,
            )
        ),
        dtype=dtype,
        device=device,
    )

    input_latent = None
    if rank == 0:
        input_latent = _build_initial_latent(args, device=device, dtype=dtype)

    LOGGER.info(
        "Simulator start rank=%s world_size=%s steps=%s backend=%s device=%s",
        rank,
        world_size,
        args.total_steps,
        backend,
        device,
    )

    try:
        final_latent = run_single_latent(
            model=model,
            total_steps=args.total_steps,
            timesteps=timesteps,
            world_size=world_size,
            rank=rank,
            latent_spec=latent_spec,
            input_latent=input_latent,
        )
        if rank == world_size - 1 and final_latent is not None:
            LOGGER.info("Final latent norm: %s", final_latent.norm().item())
    finally:
        finalize_distributed()


if __name__ == "__main__":
    main()
