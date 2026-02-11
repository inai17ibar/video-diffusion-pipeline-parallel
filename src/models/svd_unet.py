"""Stable Video Diffusion UNet wrapper for pipeline-parallel inference.

This module wraps the diffusers UNetSpatioTemporalConditionModel to match
the simple interface expected by the pipeline executor.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class StableVideoUNet(nn.Module):
    """Wrapper around UNetSpatioTemporalConditionModel for pipeline compatibility.

    This class bridges the interface gap between the pipeline's simple
    forward(latent, step) -> latent signature and the more complex
    diffusers SVD UNet interface.

    The SVD UNet expects 8-channel input:
    - 4 channels: noisy latent z_t
    - 4 channels: encoded conditioning image (repeated for each frame)

    This wrapper handles:
    - Channel concatenation for UNet input
    - Correct EulerDiscreteScheduler step (v_prediction + Karras sigmas)
    - Classifier-Free Guidance (CFG) with per-frame guidance scale
    - Shape transposition between pipeline and diffusers conventions

    Shape Convention:
        Pipeline uses (B, C, F, H, W) with C=4
        Diffusers uses (B, F, C, H, W) with C=8 (concatenated)
        This wrapper handles the transposition and concatenation automatically.

    Attributes:
        unet: The underlying UNetSpatioTemporalConditionModel
        timesteps: Full timestep schedule for step-to-timestep mapping
    """

    def __init__(
        self,
        unet: nn.Module,
        timesteps: Sequence[int],
        dtype: torch.dtype = torch.float16,
        num_train_timesteps: int = 1000,
    ) -> None:
        """Initialize the wrapper.

        Args:
            unet: Pre-loaded UNetSpatioTemporalConditionModel instance
            timesteps: Full diffusion timestep schedule (length = total_steps)
            dtype: Computation dtype (default fp16 for inference)
            num_train_timesteps: Number of training timesteps for scheduler
        """
        super().__init__()
        self.unet = unet
        self.timesteps = list(timesteps)
        self.dtype = dtype
        self.num_train_timesteps = num_train_timesteps

        # Initialize scheduler using diffusers EulerDiscreteScheduler
        self._init_scheduler()

        # Conditioning buffers - registered but not persistent
        self.register_buffer("_image_embeddings", None, persistent=False)
        self.register_buffer("_added_time_ids", None, persistent=False)
        self.register_buffer("_image_latents", None, persistent=False)
        self._conditioning_set = False

        # CFG state
        self._guidance_scale = None
        self._uncond_embeddings = None
        self._guidance_scale_tensor = None

    def _init_scheduler(self) -> None:
        """Initialize scheduler using diffusers EulerDiscreteScheduler."""
        from diffusers import EulerDiscreteScheduler

        scheduler = EulerDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=self.num_train_timesteps,
            prediction_type="v_prediction",
            interpolation_type="linear",
            timestep_spacing="leading",
            steps_offset=1,
            use_karras_sigmas=True,
        )
        scheduler.set_timesteps(len(self.timesteps))
        # sigmas: (num_steps + 1,) — last element is 0
        self.register_buffer("sigmas", scheduler.sigmas.clone(), persistent=False)
        # Official timesteps computed by the scheduler
        self.scheduler_timesteps = scheduler.timesteps.clone()
        # init_noise_sigma: scale factor for initial noise (matches diffusers)
        # For timestep_spacing="leading": (max_sigma^2 + 1)^0.5
        self._init_noise_sigma = float((self.sigmas[0] ** 2 + 1) ** 0.5)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        timesteps: Sequence[int] | None = None,
        torch_dtype: torch.dtype = torch.float16,
        enable_memory_efficient_attention: bool = True,
        enable_sliced_attention: bool = False,
        attention_slice_size: int | str = "auto",
        **kwargs,
    ) -> StableVideoUNet:
        """Load a pretrained SVD UNet and wrap it.

        Args:
            model_id: HuggingFace model ID or local path
            timesteps: Optional timestep schedule (will be generated if None)
            torch_dtype: Model loading dtype
            enable_memory_efficient_attention: Use xformers or PyTorch 2.0 attention
            enable_sliced_attention: Use attention slicing to reduce memory
            attention_slice_size: Slice size for attention ("auto", "max", or int)
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

        # Apply memory optimizations
        if enable_memory_efficient_attention:
            try:
                # Try xformers first
                unet.enable_xformers_memory_efficient_attention()
            except Exception:
                # Fallback: try to set attention backend for PyTorch 2.0
                try:
                    if hasattr(unet, "set_attention_backend"):
                        unet.set_attention_backend("flash_attention_2")
                except Exception:
                    pass  # Use default attention

        if enable_sliced_attention:
            # Note: attention slicing may not be available for all models
            try:
                if hasattr(unet, "set_attention_slice"):
                    unet.set_attention_slice(attention_slice_size)
            except Exception:
                pass  # Slicing not supported

        # Default timestep schedule if not provided
        if timesteps is None:
            # Default 25-step schedule matching Euler discrete
            timesteps = cls._default_timestep_schedule(num_steps=25)

        return cls(unet=unet, timesteps=timesteps, dtype=torch_dtype)

    def enable_memory_optimizations(self) -> None:
        """Enable memory optimizations on the UNet.

        This method tries multiple strategies:
        1. xformers memory efficient attention
        2. Flash attention backend
        3. Gradient checkpointing
        """
        # Try xformers
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            return
        except Exception:
            pass

        # Try flash attention backend
        try:
            if hasattr(self.unet, "set_attention_backend"):
                self.unet.set_attention_backend("flash_attention_2")
                return
        except Exception:
            pass

        # Enable gradient checkpointing for memory savings
        try:
            if hasattr(self.unet, "enable_gradient_checkpointing"):
                self.unet.enable_gradient_checkpointing()
        except Exception:
            pass

    @property
    def init_noise_sigma(self) -> float:
        """Scale factor for initial noise (matches diffusers convention)."""
        return self._init_noise_sigma

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
        image_latents: torch.Tensor,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        guidance_scale: float | None = None,
        num_frames: int = 14,
    ) -> None:
        """Set the conditioning inputs for video generation.

        This must be called before forward() for each new video generation.
        The conditioning persists across all diffusion steps.

        Args:
            image_embeddings: CLIP image embeddings (B, 1, 1024) or (B, 1024)
            image_latents: Encoded conditioning image (B, C, F, H, W) with C=4
            fps: Frames per second (will subtract 1 internally)
            motion_bucket_id: Motion bucket ID (typically 127)
            noise_aug_strength: Noise augmentation strength
            guidance_scale: CFG guidance scale (None or <=1.0 disables CFG)
            num_frames: Number of video frames (for per-frame guidance scale)
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
        self._image_latents = image_latents.to(self.dtype)
        self._conditioning_set = True

        # CFG setup
        self._guidance_scale = guidance_scale
        if guidance_scale is not None and guidance_scale > 1.0:
            # Unconditional: zero CLIP embeddings + zero image_latents
            self._uncond_embeddings = torch.zeros_like(self._image_embeddings)
            self._uncond_image_latents = torch.zeros_like(self._image_latents)
            # Per-frame guidance scale (linear interpolation from 1.0 to max)
            gs = torch.linspace(1.0, guidance_scale, num_frames)
            # (1, 1, F, 1, 1) shape for broadcasting with (B, C, F, H, W)
            self._guidance_scale_tensor = gs.view(1, 1, num_frames, 1, 1).to(
                device, dtype=self.dtype
            )
        else:
            self._uncond_embeddings = None
            self._uncond_image_latents = None
            self._guidance_scale_tensor = None

    def set_dummy_conditioning(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: int = 6,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        guidance_scale: float | None = None,
    ) -> None:
        """Set dummy conditioning for benchmarking purposes.

        This generates random image embeddings and latents for testing without
        requiring an actual CLIP encoder or VAE.

        Args:
            batch_size: Batch size for conditioning tensors
            num_frames: Number of video frames
            height: Latent height
            width: Latent width
            device: Device to create tensors on
            fps: Frames per second
            motion_bucket_id: Motion bucket ID
            noise_aug_strength: Noise augmentation strength
            guidance_scale: CFG guidance scale (None disables CFG)
        """
        # Generate random CLIP-like embeddings
        image_embeddings = torch.randn(
            batch_size,
            1,
            1024,
            device=device,
            dtype=self.dtype,
        )

        # Generate random image latents (4 channels, repeated for all frames)
        # Shape: (B, C=4, F, H, W)
        image_latents = torch.randn(
            batch_size,
            4,
            num_frames,
            height,
            width,
            device=device,
            dtype=self.dtype,
        )

        self.set_conditioning(
            image_embeddings=image_embeddings,
            image_latents=image_latents,
            fps=fps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            guidance_scale=guidance_scale,
            num_frames=num_frames,
        )

    def clear_conditioning(self) -> None:
        """Clear the conditioning state."""
        self._image_embeddings = None
        self._added_time_ids = None
        self._image_latents = None
        self._conditioning_set = False
        self._guidance_scale = None
        self._uncond_embeddings = None
        self._uncond_image_latents = None
        self._guidance_scale_tensor = None

    @torch.inference_mode()
    def forward(self, latent: torch.Tensor, step: int) -> torch.Tensor:
        """Execute a single diffusion step with scheduler update.

        Uses v_prediction with correct scale_model_input and optional CFG.

        Args:
            latent: Noisy latent tensor (B, C=4, F, H, W)
            step: Step index into the timestep schedule

        Returns:
            Denoised latent tensor (B, C=4, F, H, W)

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
            raise ValueError(f"Step {step} out of range [0, {len(self.timesteps)})")

        sigma = self.sigmas[step]
        sigma_next = self.sigmas[step + 1]
        timestep = self.scheduler_timesteps[step]

        # 1. Scale model input: sample / sqrt(sigma^2 + 1)
        latent_scaled = latent / ((sigma**2 + 1) ** 0.5)

        # 2. Classifier-Free Guidance (sequential for memory efficiency)
        if self._guidance_scale is not None and self._guidance_scale > 1.0:
            # Unconditional pass: zero image_latents + zero CLIP embeddings
            uncond_input = torch.cat([latent_scaled, self._uncond_image_latents], dim=1)
            uncond_sample = uncond_input.permute(0, 2, 1, 3, 4).to(self.dtype)
            noise_pred_uncond = self.unet(
                sample=uncond_sample,
                timestep=timestep,
                encoder_hidden_states=self._uncond_embeddings,
                added_time_ids=self._added_time_ids,
                return_dict=False,
            )[0]

            # Conditional pass: real image_latents + real CLIP embeddings
            cond_input = torch.cat([latent_scaled, self._image_latents], dim=1)
            cond_sample = cond_input.permute(0, 2, 1, 3, 4).to(self.dtype)
            noise_pred_cond = self.unet(
                sample=cond_sample,
                timestep=timestep,
                encoder_hidden_states=self._image_embeddings,
                added_time_ids=self._added_time_ids,
                return_dict=False,
            )[0]

            # Apply guidance (per-frame scale)
            # noise_pred: (B, F, C=4, H, W), guidance_scale_tensor needs permute
            gs = self._guidance_scale_tensor.permute(0, 2, 1, 3, 4)  # (1, F, 1, 1, 1)
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
        else:
            # No CFG: real image_latents + real CLIP embeddings
            latent_model_input = torch.cat([latent_scaled, self._image_latents], dim=1)
            sample = latent_model_input.permute(0, 2, 1, 3, 4).to(self.dtype)
            noise_pred = self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=self._image_embeddings,
                added_time_ids=self._added_time_ids,
                return_dict=False,
            )[0]

        # 5. Transpose back: (B, F, C=4, H, W) -> (B, C=4, F, H, W)
        noise_pred = noise_pred.permute(0, 2, 1, 3, 4)

        # 6. V-prediction Euler step (fp32 for numerical stability)
        latent_f32 = latent.float()
        noise_pred_f32 = noise_pred.float()
        sigma_f32 = sigma.float()

        pred_original_sample = noise_pred_f32 * (
            -sigma_f32 / (sigma_f32**2 + 1) ** 0.5
        ) + latent_f32 / (sigma_f32**2 + 1)
        derivative = (latent_f32 - pred_original_sample) / sigma_f32
        dt = float(sigma_next) - float(sigma)
        prev_sample = latent_f32 + derivative * dt

        return prev_sample.to(self.dtype)
