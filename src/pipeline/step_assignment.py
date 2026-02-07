"""Step assignment utilities for pipeline parallel inference.

This module intentionally avoids importing ``torch.distributed`` so that logic can
be unit-tested without initializing any distributed context.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepRange:
    """Closed-open interval of diffusion steps assigned to a rank."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Step indices must be non-negative.")
        if self.end < self.start:
            raise ValueError("Step range end must be >= start.")

    @property
    def count(self) -> int:
        """Number of steps represented by the range."""

        return self.end - self.start

    def __iter__(self):  # pragma: no cover - convenience helper
        yield from range(self.start, self.end)


def assign_steps(total_steps: int, world_size: int, rank: int) -> StepRange:
    """Compute the diffusion step range for a given rank.

    Args:
        total_steps: Total number of diffusion steps in the schedule.
        world_size: Number of pipeline stages / distributed ranks.
        rank: Rank for which we want the step allocation (0-indexed).

    Returns:
        ``StepRange`` describing the closed-open interval [start, end) handled by
        ``rank``.

    Raises:
        ValueError: If the input arguments are inconsistent or if ``total_steps`` is
            not divisible by ``world_size``. The AGENTS.md document currently
            mandates equal, contiguous splits so we surface invalid schedules early.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if not (0 <= rank < world_size):
        raise ValueError("rank must satisfy 0 <= rank < world_size.")

    if total_steps % world_size != 0:
        raise ValueError(
            "total_steps must be divisible by world_size for uniform step assignment."
        )

    steps_per_rank = total_steps // world_size
    start_step = rank * steps_per_rank
    end_step = start_step + steps_per_rank

    return StepRange(start=start_step, end=end_step)
