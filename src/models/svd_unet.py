"""Stable Video Diffusion UNet wrapper for pipeline-parallel inference.

This module wraps the diffusers UNetSpatioTemporalConditionModel to match
the simple interface expected by the pipeline executor.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


class StableVideoUNet(nn.Module):
    """Wrapper around UNetSpatioTemporalConditionModel for pipeline compatibility.

    This class bridges the interface gap between the pipeline's simple
    forward(latent, step) -> latent signature and the more complex
    diffusers SVD UNet interface.

    Shape Convention:
        Pipeline uses (B, C, F, H, W)
        Diffusers uses (B, F, C, H, W)
        This wrapper handles the transposition automatically.

    Attributes:
        unet: The underlying UNetSpatioTemporalConditionModel
        timesteps: Full timestep schedule for step-to-timestep mapping
    """

    def __init__(
        self,
        unet: nn.Module,
        timesteps: Sequence[int],
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize the wrapper.

        Args:
            unet: Pre-loaded UNetSpatioTemporalConditionModel instance
            timesteps: Full diffusion timestep schedule (length = total_steps)
            dtype: Computation dtype (default fp16 for inference)
        """
        super().__init__()
        self.unet = unet
        self.timesteps = list(timesteps)
        self.dtype = dtype

        # Conditioning buffers - registered but not persistent
        self.register_buffer("_image_embeddings", None, persistent=False)
        self.register_buffer("_added_time_ids", None, persistent=False)
        self._conditioning_set = False

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        timesteps: Optional[Sequence[int]] = None,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "StableVideoUNet":
        """Load a pretrained SVD UNet and wrap it.

        Args:
            model_id: HuggingFace model ID or local path
            timesteps: Optional timestep schedule (will be generated if None)
            torch_dtype: Model loading dtype
            **kwargs: Additional arguments passed to from_pretrained

        Returns:
            StableVideoUNet instance ready for pipeline use
        """
        from diffusers.models import UNetSpatioTemporalConditionModel

        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            torch_dtype=torch_dtype,
            **kwargs,
        )

        # Default timestep schedule if not provided
        if timesteps is None:
            # Default 25-step schedule matching Euler discrete
            timesteps = cls._default_timestep_schedule(num_steps=25)

        return cls(unet=unet, timesteps=timesteps, dtype=torch_dtype)

    @staticmethod
    def _default_timestep_schedule(
        num_steps: int,
        num_train_timesteps: int = 1000,
    ) -> list[int]:
        """Generate default Euler discrete timestep schedule.

        Args:
            num_steps: Number of inference steps
            num_train_timesteps: Training timestep range

        Returns:
            List of timesteps in descending order
        """
        step_ratio = num_train_timesteps // num_steps
        timesteps = list(range(num_train_timesteps - 1, -1, -step_ratio))[:num_steps]
        return timesteps

    def set_conditioning(
        self,
        image_embeddings: torch.Tensor,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
    ) -> None:
        """Set the conditioning inputs for video generation.

        This must be called before forward() for each new video generation.
        The conditioning persists across all diffusion steps.

        Args:
            image_embeddings: CLIP image embeddings (B, 1, 1024) or (B, 1024)
            fps: Frames per second (will subtract 1 internally)
            motion_bucket_id: Motion bucket ID (typically 127)
            noise_aug_strength: Noise augmentation strength
        """
        # Ensure 3D embeddings: (B, seq_len, embed_dim)
        if image_embeddings.dim() == 2:
            image_embeddings = image_embeddings.unsqueeze(1)

        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device

        # Build added_time_ids: (B, 3)
        # Values: [fps - 1, motion_bucket_id, noise_aug_strength]
        added_time_ids = torch.tensor(
            [[fps - 1, motion_bucket_id, noise_aug_strength]],
            dtype=self.dtype,
            device=device,
        ).repeat(batch_size, 1)

        # Register as buffers for proper device handling
        self._image_embeddings = image_embeddings.to(self.dtype)
        self._added_time_ids = added_time_ids
        self._conditioning_set = True

    def set_dummy_conditioning(
        self,
        batch_size: int,
        device: torch.device,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
    ) -> None:
        """Set dummy conditioning for benchmarking purposes.

        This generates random image embeddings for testing without
        requiring an actual CLIP encoder.

        Args:
            batch_size: Batch size for conditioning tensors
            device: Device to create tensors on
            fps: Frames per second
            motion_bucket_id: Motion bucket ID
            noise_aug_strength: Noise augmentation strength
        """
        # Generate random CLIP-like embeddings
        image_embeddings = torch.randn(
            batch_size, 1, 1024,
            device=device,
            dtype=self.dtype,
        )
        self.set_conditioning(
            image_embeddings=image_embeddings,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        )

    def clear_conditioning(self) -> None:
        """Clear the conditioning state."""
        self._image_embeddings = None
        self._added_time_ids = None
        self._conditioning_set = False

    def forward(self, latent: torch.Tensor, step: int) -> torch.Tensor:
        """Execute a single diffusion step.

        Args:
            latent: Noisy latent tensor (B, C, F, H, W)
            step: Step index into the timestep schedule

        Returns:
            Denoised latent tensor (B, C, F, H, W)

        Raises:
            RuntimeError: If set_conditioning() was not called
        """
        if not self._conditioning_set:
            raise RuntimeError(
                "Conditioning not set. Call set_conditioning() or "
                "set_dummy_conditioning() before forward()."
            )

        # Validate step index
        if not (0 <= step < len(self.timesteps)):
            raise ValueError(
                f"Step {step} out of range [0, {len(self.timesteps)})"
            )

        # Map step index to actual timestep value
        timestep = self.timesteps[step]

        # Transpose: (B, C, F, H, W) -> (B, F, C, H, W)
        sample = latent.permute(0, 2, 1, 3, 4).to(self.dtype)

        # Forward through the actual UNet
        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=self._image_embeddings,
            added_time_ids=self._added_time_ids,
            return_dict=False,
        )[0]

        # Transpose back: (B, F, C, H, W) -> (B, C, F, H, W)
        result = output.permute(0, 2, 1, 3, 4)

        return result
