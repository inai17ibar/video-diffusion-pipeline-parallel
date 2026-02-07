"""Lightweight UNet stand-in for simulator mode.

The goal is to exercise pipeline control flow, not achieve any particular
quality. The module performs a couple of inexpensive 3D convolutions and mixes in
step-dependent scaling so that sequential ordering matters during debugging.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class DummyUNet(nn.Module):
    def __init__(
        self,
        channels: int = 8,
        hidden_channels: int = 16,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv3d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, channels, kernel_size=3, padding=1),
        ]
        self.net = nn.Sequential(*layers)
        self.norm: Optional[nn.Module]
        if use_layernorm:
            self.norm = nn.LayerNorm(normalized_shape=channels)
        else:
            self.norm = None

    def forward(self, latent: torch.Tensor, step: int) -> torch.Tensor:  # type: ignore[override]
        residual = latent
        out = self.net(latent)
        scale = math.tanh(step / 10.0)
        out = residual + scale * out
        if self.norm is not None:
            # LayerNorm expects channels on the last dim. Move C to the end,
            # apply LN, then restore the original ordering.
            dims = residual.dim()
            if dims < 2:
                raise ValueError("Latent tensor must have at least 2 dims (N, C, ...)")
            to_last = list(range(dims))
            to_last.append(1)
            to_last.pop(1)
            from_last = [0] * dims
            for idx, axis in enumerate(to_last):
                from_last[axis] = idx

            permuted = residual.permute(*to_last)
            normalized = self.norm(permuted)
            restored = normalized.permute(*from_last)
            out = out + restored
        return out
