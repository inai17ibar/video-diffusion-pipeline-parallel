"""Backend selection helpers for distributed modes."""

from __future__ import annotations

import os
from typing import Optional

_SUPPORTED = {"nccl", "gloo"}
_DEFAULT_ENV_VAR = "PIPELINE_BACKEND"


def resolve_backend(preferred: Optional[str] = None, *, simulator: bool = False) -> str:
    """Return the backend string to pass to ``init_process_group``.

    Order of precedence:
    1. ``preferred`` argument if provided.
    2. ``PIPELINE_BACKEND`` env var (override for debugging).
    3. "gloo" when ``simulator`` is True else "nccl".
    """

    if preferred:
        backend = preferred.lower()
    else:
        backend = os.environ.get(_DEFAULT_ENV_VAR, "").lower()

    if backend:
        if backend not in _SUPPORTED:
            raise ValueError(f"Unsupported backend '{backend}'.")
        return backend

    return "gloo" if simulator else "nccl"
