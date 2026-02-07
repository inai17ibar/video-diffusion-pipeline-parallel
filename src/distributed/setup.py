"""Utilities for initializing/destroying torch.distributed."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Optional

import torch.distributed as dist

LOGGER = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = timedelta(minutes=10)


def init_distributed(
    *,
    backend: str,
    rank: int,
    world_size: int,
    init_method: Optional[str] = None,
    timeout: Optional[timedelta] = None,
) -> None:
    """Initialize the default process group if not already done."""

    if dist.is_initialized():
        LOGGER.debug("Process group already initialized.")
        return

    kwargs = {
        "backend": backend,
        "rank": rank,
        "world_size": world_size,
        "timeout": timeout or _DEFAULT_TIMEOUT,
    }
    if init_method:
        kwargs["init_method"] = init_method

    LOGGER.info(
        "Initializing process group backend=%s rank=%s world_size=%s", backend, rank, world_size
    )
    dist.init_process_group(**kwargs)


def finalize_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
