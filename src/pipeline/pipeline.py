"""Minimal pipeline-parallel executor for diffusion inference.

The design is intentionally small in scope for the initial milestone:
- One latent sample flows through the pipeline (no fill / drain overlap).
- Each rank executes a contiguous chunk of diffusion steps.
- Communication happens via explicit ``dist.send`` / ``dist.recv`` calls.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from .step_assignment import StepRange, assign_steps

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatentSpec:
    """Description of the latent tensor shape each rank expects to handle."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device

    def empty(self) -> torch.Tensor:
        return torch.empty(self.shape, dtype=self.dtype, device=self.device)


@dataclass(frozen=True)
class PipelineConfig:
    total_steps: int
    world_size: int
    rank: int
    timesteps: Sequence[int]
    latent_spec: LatentSpec
    send_tag: int = 0

    def __post_init__(self) -> None:
        if len(self.timesteps) != self.total_steps:
            raise ValueError("len(timesteps) must equal total_steps.")


InputSupplier = Callable[[int], torch.Tensor]


class PipelineStage:
    """Owns the execution + send/recv behavior for a single rank."""

    def __init__(
        self,
        model,
        config: PipelineConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.logger = logger or LOGGER
        self.step_range: StepRange = assign_steps(
            total_steps=config.total_steps,
            world_size=config.world_size,
            rank=config.rank,
        )

    def _log(self, message: str) -> None:
        self.logger.info("[rank=%s] %s", self.config.rank, message)

    def _recv_latent(self) -> torch.Tensor:
        tensor = self.config.latent_spec.empty()
        self._log(f"waiting for latent from rank {self.config.rank - 1}")
        dist.recv(tensor, src=self.config.rank - 1, tag=self.config.send_tag)
        self._log("received latent")
        return tensor

    def _send_latent(self, latent: torch.Tensor) -> None:
        self._log(f"sending latent to rank {self.config.rank + 1}")
        dist.send(latent, dst=self.config.rank + 1, tag=self.config.send_tag)

    def _run_local_steps(self, latent: torch.Tensor) -> torch.Tensor:
        local_timesteps: list[int] = list(
            self.config.timesteps[self.step_range.start : self.step_range.end]
        )
        if len(local_timesteps) != self.step_range.count:
            raise RuntimeError("Local timestep slice length mismatch with step range.")

        for step in local_timesteps:
            start_time = time.time()
            latent = self.model(latent, step)
            elapsed_ms = (time.time() - start_time) * 1000.0
            self._log(f"step {step} completed in {elapsed_ms:.2f} ms")
        return latent

    def run(self, input_latent: torch.Tensor | None) -> torch.Tensor | None:
        """Execute the stage and forward the latent downstream.

        Args:
            input_latent: Only rank 0 should pass an initial latent tensor. Other
                ranks must pass ``None`` and will block on ``recv``.

        Returns:
            The final latent tensor if this is the last rank, otherwise ``None``.
        """

        return self._process_single_latent(input_latent, sample_idx=None)

    def run_many(
        self,
        num_samples: int,
        *,
        input_supplier: InputSupplier | None = None,
    ) -> list[torch.Tensor] | None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive for pipeline execution")
        if self.config.rank == 0 and input_supplier is None:
            raise ValueError("rank 0 requires an input_supplier when processing multiple samples")
        outputs: list[torch.Tensor] = []
        for sample_idx in range(num_samples):
            latent = self._process_single_latent(
                input_supplier(sample_idx) if self.config.rank == 0 else None,
                sample_idx=sample_idx,
            )
            if latent is not None:
                outputs.append(latent)

        return outputs if outputs else None

    def _process_single_latent(
        self, input_latent: torch.Tensor | None, sample_idx: int | None
    ) -> torch.Tensor | None:
        prefix = f"sample {sample_idx} " if sample_idx is not None else ""

        if self.config.rank == 0:
            if input_latent is None:
                raise ValueError("rank 0 requires an input latent tensor")
            latent = input_latent.to(self.config.latent_spec.device)
            self._log(f"{prefix}input prepared")
        else:
            if input_latent is not None:
                raise ValueError("non-zero ranks should not receive an eager latent")
            latent = self._recv_latent()
            self._log(f"{prefix}received latent")

        latent = self._run_local_steps(latent)

        if self.config.rank == self.config.world_size - 1:
            self._log(f"{prefix}final rank completed")
            return latent

        self._send_latent(latent)
        return None


def run_single_latent(
    model,
    *,
    total_steps: int,
    timesteps: Sequence[int],
    world_size: int,
    rank: int,
    latent_spec: LatentSpec,
    input_latent: torch.Tensor | None,
    logger: logging.Logger | None = None,
) -> torch.Tensor | None:
    """Convenience entrypoint used by mode scripts.

    All ranks should call this function once per latent. Rank 0 passes the
    ``input_latent`` it generated. Other ranks must pass ``None``.
    """

    config = PipelineConfig(
        total_steps=total_steps,
        world_size=world_size,
        rank=rank,
        timesteps=timesteps,
        latent_spec=latent_spec,
    )
    stage = PipelineStage(model=model, config=config, logger=logger)
    return stage.run(input_latent=input_latent)


def run_pipeline_latents(
    model,
    *,
    total_steps: int,
    timesteps: Sequence[int],
    world_size: int,
    rank: int,
    latent_spec: LatentSpec,
    num_samples: int,
    input_supplier: InputSupplier | None,
    logger: logging.Logger | None = None,
) -> list[torch.Tensor] | None:
    config = PipelineConfig(
        total_steps=total_steps,
        world_size=world_size,
        rank=rank,
        timesteps=timesteps,
        latent_spec=latent_spec,
    )
    stage = PipelineStage(model=model, config=config, logger=logger)
    return stage.run_many(num_samples, input_supplier=input_supplier)
