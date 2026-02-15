#!/usr/bin/env python3
"""LTデモ用: 実際の画像から動画を生成するスクリプト.

このスクリプトは、入力画像から実際の動画を生成し、
パイプライン並列の効果を視覚的にデモンストレーションします。

使用方法（単一GPU）:
    python scripts/generate_video_demo.py --input-image path/to/image.jpg

使用方法（マルチGPU）:
    torchrun --nproc_per_node=7 scripts/generate_video_demo.py \
        --input-image path/to/image.jpg \
        --total-steps 21

必要な依存関係:
    pip install diffusers transformers accelerate pillow imageio
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate video from image using pipeline parallel"
    )
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--total-steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--num-frames", type=int, default=14, help="Number of video frames")
    parser.add_argument("--fps", type=int, default=7, help="Output video FPS")
    parser.add_argument(
        "--motion-bucket-id", type=int, default=127, help="Motion bucket ID (0-255)"
    )
    parser.add_argument(
        "--noise-aug-strength", type=float, default=0.02, help="Noise augmentation strength"
    )
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=576, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument(
        "--guidance-scale", type=float, default=3.0, help="CFG guidance scale (1.0 disables CFG)"
    )
    parser.add_argument(
        "--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def discover_distributed_info() -> tuple[int, int, bool]:
    """Discover rank and world_size from environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    return rank, world_size, local_rank, is_distributed


def load_and_preprocess_image(image_path: str, height: int, width: int):
    """Load and preprocess input image using center crop (no aspect ratio distortion)."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    src_w, src_h = image.size

    # Scale so the shorter side matches the target, then center crop
    scale = max(width / src_w, height / src_h)
    new_w = round(src_w * scale)
    new_h = round(src_h * scale)
    if (new_w, new_h) != (src_w, src_h):
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # Center crop to exact target size
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    image = image.crop((left, top, left + width, top + height))
    return image


def encode_image(
    image,
    image_encoder: torch.nn.Module,
    feature_extractor,
    vae: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    num_frames: int,
    noise_aug_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode image to CLIP embeddings and VAE latents.

    Follows the official diffusers SVDPipeline convention:
    - CLIP embeddings: standard image encoding
    - VAE latents: encode with noise augmentation in pixel space, NO scaling_factor,
      use .mode() for deterministic encoding
    """
    # CLIP image encoding
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device, dtype=dtype)

    with torch.no_grad():
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)  # (B, 1, 1024)

    # VAE encoding (matching official diffusers pipeline)
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    # Add noise augmentation in PIXEL space (before VAE encoding)
    if noise_aug_strength > 0:
        noise = torch.randn_like(image_tensor)
        image_tensor = image_tensor + noise_aug_strength * noise

    # Upcast VAE to fp32 if force_upcast is enabled (matches official diffusers pipeline)
    needs_upcast = getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16
    if needs_upcast:
        vae.to(dtype=torch.float32)
        image_tensor = image_tensor.to(dtype=torch.float32)

    with torch.no_grad():
        # Use .mode() for deterministic encoding (official pipeline convention)
        # Do NOT multiply by scaling_factor — image_latents are raw VAE latents
        image_latents = vae.encode(image_tensor).latent_dist.mode()

    if needs_upcast:
        vae.to(dtype=torch.float16)
        image_latents = image_latents.to(dtype=dtype)

    # Repeat for all frames: (B, C, H, W) -> (B, C, F, H, W)
    image_latents = image_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

    return image_embeddings, image_latents


def decode_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    num_frames: int,
    decode_chunk_size: int = 14,
) -> torch.Tensor:
    """Decode latents to video frames.

    Follows the official diffusers SVDPipeline convention:
    - Decode all frames at once (or in chunks) so the temporal decoder can work
    - Apply 1/scaling_factor before decoding
    """
    # latents: (B, C, F, H, W) -> (B, F, C, H, W) -> (B*F, C, H, W)
    latents = latents.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)
    batch_size = latents.shape[0]
    latents = latents.flatten(0, 1)  # (B*F, C, H, W)
    latents = latents / vae.config.scaling_factor

    # Upcast VAE to fp32 if force_upcast is enabled (matches official diffusers pipeline)
    needs_upcast = getattr(vae.config, "force_upcast", False) and vae.dtype == torch.float16
    if needs_upcast:
        vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)

    # Decode in chunks (all frames together for temporal coherence)
    frames = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], decode_chunk_size):
            chunk = latents[i : i + decode_chunk_size]
            frame = vae.decode(chunk, num_frames=chunk.shape[0]).sample
            frames.append(frame)
    frames = torch.cat(frames, dim=0)  # (B*F, C_out, H_pixel, W_pixel)

    if needs_upcast:
        vae.to(dtype=torch.float16)

    # Reshape: (B*F, C, H, W) -> (B, F, C, H, W) -> (B, C, F, H, W)
    frames = frames.reshape(batch_size, num_frames, *frames.shape[1:])
    frames = frames.permute(0, 2, 1, 3, 4)

    frames = frames.float()
    return frames


def save_video(frames: torch.Tensor, output_path: str, fps: int) -> None:
    """Save video frames to file."""
    import imageio

    # frames: (B, C, F, H, W) -> (F, H, W, C) for first batch
    frames = frames[0]  # Remove batch dim
    frames = frames.permute(1, 2, 3, 0)  # (F, H, W, C)
    frames = ((frames + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Save as MP4
    imageio.mimsave(output_path, frames, fps=fps)
    LOGGER.info(f"Video saved to: {output_path}")


def save_gif(frames: torch.Tensor, output_path: str, fps: int) -> None:
    """Save video frames as GIF."""
    import imageio

    frames = frames[0]  # Remove batch dim
    frames = frames.permute(1, 2, 3, 0)  # (F, H, W, C)
    frames = ((frames + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Save as GIF
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    LOGGER.info(f"GIF saved to: {output_path}")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rank, world_size, local_rank, is_distributed = discover_distributed_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    LOGGER.info(f"Rank {rank}/{world_size}, Device: {device}")

    # Initialize distributed if needed
    if is_distributed:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        LOGGER.info(f"Distributed initialized: rank={rank}, world_size={world_size}")

    # Load models (only need full pipeline on rank 0 for encoding/decoding)
    LOGGER.info("Loading models...")
    start_load = time.time()

    from diffusers import AutoencoderKLTemporalDecoder
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    # Load image encoder and VAE on all ranks for encoding
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.model_id, subfolder="image_encoder", torch_dtype=dtype
    ).to(device)
    image_encoder.eval()

    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.model_id, subfolder="feature_extractor"
    )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=dtype
    ).to(device)
    vae.eval()

    # Load UNet with our wrapper
    from src.models.svd_unet import StableVideoUNet

    timesteps = StableVideoUNet._default_timestep_schedule(args.total_steps)
    model = StableVideoUNet.from_pretrained(
        model_id=args.model_id,
        timesteps=timesteps,
        torch_dtype=dtype,
        enable_memory_efficient_attention=True,
        enable_sliced_attention=True,
        attention_slice_size="auto",
    ).to(device)
    model.enable_memory_optimizations()

    load_time = time.time() - start_load
    LOGGER.info(f"Models loaded in {load_time:.2f}s")

    # Load and encode input image
    LOGGER.info(f"Loading input image: {args.input_image}")
    image = load_and_preprocess_image(args.input_image, args.height, args.width)

    # Calculate latent dimensions
    latent_height = args.height // 8
    latent_width = args.width // 8

    LOGGER.info("Encoding input image...")
    image_embeddings, image_latents = encode_image(
        image=image,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        vae=vae,
        device=device,
        dtype=dtype,
        num_frames=args.num_frames,
        noise_aug_strength=args.noise_aug_strength,
    )

    # Free CLIP encoder (no longer needed after encoding)
    del image_encoder, feature_extractor
    # Free VAE on non-final ranks (only final rank needs it for decoding)
    if is_distributed and rank != world_size - 1:
        del vae
    torch.cuda.empty_cache()
    LOGGER.info("Freed encoder models, CUDA cache cleared")

    # Set conditioning on model (with CFG)
    model.set_conditioning(
        image_embeddings=image_embeddings,
        image_latents=image_latents,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        guidance_scale=args.guidance_scale,
        num_frames=args.num_frames,
    )

    # Compute step assignment once (reused for all samples)
    if is_distributed:
        from src.pipeline.step_assignment import assign_steps

        step_range = assign_steps(args.total_steps, world_size, rank)
        LOGGER.info(f"Rank {rank}: processing steps {step_range.start} to {step_range.end - 1}")

    # Prepare output directory and naming
    output_dir = Path(args.output_dir)
    if rank == world_size - 1:
        output_dir.mkdir(parents=True, exist_ok=True)
    input_name = Path(args.input_image).stem
    timestamp = int(time.time())

    num_samples = args.num_samples
    total_diffusion_time = 0.0
    total_decode_time = 0.0
    # Store completed latents on CPU and their seed info for later decoding
    sample_results: list[tuple[torch.Tensor, int, int]] = []  # (latents_cpu, sample_idx, seed)

    LOGGER.info(f"Generating {num_samples} samples...")

    # Phase 1: Run diffusion for all samples (UNet on GPU)
    for sample_idx in range(num_samples):
        sample_seed = args.seed + sample_idx
        LOGGER.info(f"--- Sample {sample_idx + 1}/{num_samples} (seed={sample_seed}) ---")

        # Generate initial noise on rank 0, broadcast to others via send/recv
        latent_shape = (1, 4, args.num_frames, latent_height, latent_width)
        if is_distributed:
            if rank == 0:
                torch.manual_seed(sample_seed)
                latents = torch.randn(*latent_shape, device=device, dtype=dtype)
                latents = latents * model.init_noise_sigma
            else:
                latents = torch.empty(*latent_shape, device=device, dtype=dtype)
        else:
            torch.manual_seed(sample_seed)
            latents = torch.randn(*latent_shape, device=device, dtype=dtype)
            latents = latents * model.init_noise_sigma

        LOGGER.info(f"Initial noise scaled by init_noise_sigma={model.init_noise_sigma:.4f}")

        # Run diffusion steps
        LOGGER.info(
            f"Running {args.total_steps} diffusion steps (guidance_scale={args.guidance_scale})..."
        )
        start_diffusion = time.time()

        if is_distributed:
            # Receive from previous rank (if not first)
            if rank > 0:
                dist.recv(latents, src=rank - 1)
                LOGGER.info(f"Rank {rank}: received latents from rank {rank - 1}")

            # Process local steps
            for step_idx in range(step_range.start, step_range.end):
                step_start = time.time()
                latents = model(latents, step_idx)
                torch.cuda.empty_cache()
                step_time = (time.time() - step_start) * 1000
                LOGGER.info(f"Rank {rank}: step {step_idx} completed in {step_time:.2f}ms")

            # Send to next rank (if not last)
            if rank < world_size - 1:
                dist.send(latents, dst=rank + 1)
                LOGGER.info(f"Rank {rank}: sent latents to rank {rank + 1}")

        else:
            # Single GPU execution
            for step_idx in range(args.total_steps):
                step_start = time.time()
                latents = model(latents, step_idx)
                torch.cuda.empty_cache()
                step_time = (time.time() - step_start) * 1000
                if step_idx % 5 == 0 or step_idx == args.total_steps - 1:
                    LOGGER.info(
                        f"Step {step_idx}/{args.total_steps} completed in {step_time:.2f}ms"
                    )

        diffusion_time = time.time() - start_diffusion
        total_diffusion_time += diffusion_time
        LOGGER.info(f"Diffusion completed in {diffusion_time:.2f}s")

        # Save latents to CPU for later decoding (only on final rank)
        if rank == world_size - 1:
            sample_results.append((latents.cpu(), sample_idx, sample_seed))

    # Phase 2: Free UNet, then decode all samples with VAE
    del model
    torch.cuda.empty_cache()
    LOGGER.info("Freed UNet model for decoding")

    if rank == world_size - 1:
        for latents_cpu, sample_idx, sample_seed in sample_results:
            LOGGER.info(f"--- Decoding sample {sample_idx + 1}/{num_samples} ---")
            start_decode = time.time()
            video_frames = decode_latents(
                latents_cpu.to(device), vae, args.num_frames, decode_chunk_size=4
            )
            decode_time = time.time() - start_decode
            total_decode_time += decode_time
            LOGGER.info(f"Decoding completed in {decode_time:.2f}s")

            # Save MP4
            mp4_name = (
                f"{input_name}_svd_{world_size}gpu_s{sample_idx}_seed{sample_seed}_{timestamp}.mp4"
            )
            mp4_path = output_dir / mp4_name
            save_video(video_frames, str(mp4_path), args.fps)

            # Save GIF
            gif_name = (
                f"{input_name}_svd_{world_size}gpu_s{sample_idx}_seed{sample_seed}_{timestamp}.gif"
            )
            gif_path = output_dir / gif_name
            save_gif(video_frames, str(gif_path), args.fps)

    # Save input image for comparison (once, after all samples)
    if rank == world_size - 1:
        input_copy_path = output_dir / f"{input_name}_input_{timestamp}.png"
        image.save(str(input_copy_path))
        LOGGER.info(f"Input image saved to: {input_copy_path}")

        # Print summary
        total_time = load_time + total_diffusion_time + total_decode_time
        LOGGER.info("=" * 50)
        LOGGER.info("Generation Summary")
        LOGGER.info("=" * 50)
        LOGGER.info(f"  Input: {args.input_image}")
        LOGGER.info(f"  Output dir: {output_dir}")
        LOGGER.info(f"  Samples: {num_samples}")
        LOGGER.info(f"  GPUs: {world_size}")
        LOGGER.info(f"  Steps: {args.total_steps}")
        LOGGER.info(f"  Frames: {args.num_frames}")
        LOGGER.info(f"  Resolution: {args.width}x{args.height}")
        LOGGER.info(f"  Guidance scale: {args.guidance_scale}")
        LOGGER.info(f"  Model load time: {load_time:.2f}s")
        LOGGER.info(f"  Total diffusion time: {total_diffusion_time:.2f}s")
        LOGGER.info(f"  Total decode time: {total_decode_time:.2f}s")
        LOGGER.info(f"  Total time: {total_time:.2f}s")
        LOGGER.info("=" * 50)

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
