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
    parser = argparse.ArgumentParser(description="Generate video from image using pipeline parallel")
    parser.add_argument("--input-image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--total-steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--num-frames", type=int, default=14, help="Number of video frames")
    parser.add_argument("--fps", type=int, default=7, help="Output video FPS")
    parser.add_argument("--motion-bucket-id", type=int, default=127, help="Motion bucket ID (0-255)")
    parser.add_argument("--noise-aug-strength", type=float, default=0.02, help="Noise augmentation strength")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--height", type=int, default=576, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def discover_distributed_info() -> tuple[int, int, bool]:
    """Discover rank and world_size from environment."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    return rank, world_size, local_rank, is_distributed


def load_and_preprocess_image(image_path: str, height: int, width: int) -> "Image.Image":
    """Load and preprocess input image."""
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((width, height), Image.LANCZOS)
    return image


def encode_image(
    image: "Image.Image",
    image_encoder: torch.nn.Module,
    feature_extractor,
    vae: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    num_frames: int,
    noise_aug_strength: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode image to CLIP embeddings and VAE latents."""
    import torch.nn.functional as F

    # CLIP image encoding
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device, dtype=dtype)

    with torch.no_grad():
        image_embeddings = image_encoder(pixel_values).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)  # (B, 1, 1024)

    # VAE encoding
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        image_latents = vae.encode(image_tensor).latent_dist.sample()
        image_latents = image_latents * vae.config.scaling_factor

    # Add noise augmentation
    if noise_aug_strength > 0:
        noise = torch.randn_like(image_latents)
        image_latents = image_latents + noise_aug_strength * noise

    # Repeat for all frames: (B, C, H, W) -> (B, C, F, H, W)
    image_latents = image_latents.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

    return image_embeddings, image_latents


def decode_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    num_frames: int,
) -> torch.Tensor:
    """Decode latents to video frames."""
    # latents: (B, C, F, H, W) -> decode each frame
    latents = latents / vae.config.scaling_factor

    frames = []
    for i in range(num_frames):
        frame_latent = latents[:, :, i, :, :]  # (B, C, H, W)
        with torch.no_grad():
            frame = vae.decode(frame_latent).sample
        frames.append(frame)

    # Stack frames: (B, C, F, H, W)
    video = torch.stack(frames, dim=2)
    return video


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
    from diffusers.models import UNetSpatioTemporalConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    # Load image encoder and VAE on rank 0 (or all ranks for simplicity)
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
    ).to(device)

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

    # Set conditioning on model
    model.set_conditioning(
        image_embeddings=image_embeddings,
        image_latents=image_latents,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
    )

    # Generate initial noise
    torch.manual_seed(args.seed)
    latents = torch.randn(
        1, 4, args.num_frames, latent_height, latent_width,
        device=device, dtype=dtype
    )

    # Run diffusion steps
    LOGGER.info(f"Running {args.total_steps} diffusion steps...")
    start_diffusion = time.time()

    if is_distributed:
        # Pipeline parallel execution
        from src.pipeline.step_assignment import assign_steps

        step_range = assign_steps(args.total_steps, world_size, rank)
        LOGGER.info(f"Rank {rank}: processing steps {step_range.start} to {step_range.end - 1}")

        # Receive from previous rank (if not first)
        if rank > 0:
            latents = torch.empty_like(latents)
            dist.recv(latents, src=rank - 1)
            LOGGER.info(f"Rank {rank}: received latents from rank {rank - 1}")

        # Process local steps
        for step_idx in range(step_range.start, step_range.end):
            step = timesteps[step_idx] if step_idx < len(timesteps) else step_idx
            step_start = time.time()
            latents = model(latents, step_idx)
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
            step_time = (time.time() - step_start) * 1000
            if step_idx % 5 == 0 or step_idx == args.total_steps - 1:
                LOGGER.info(f"Step {step_idx}/{args.total_steps} completed in {step_time:.2f}ms")

    diffusion_time = time.time() - start_diffusion
    LOGGER.info(f"Diffusion completed in {diffusion_time:.2f}s")

    # Only last rank (or single GPU) decodes and saves
    if rank == world_size - 1:
        LOGGER.info("Decoding latents to video frames...")
        start_decode = time.time()
        video_frames = decode_latents(latents, vae, args.num_frames)
        decode_time = time.time() - start_decode
        LOGGER.info(f"Decoding completed in {decode_time:.2f}s")

        # Save output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_name = Path(args.input_image).stem
        timestamp = int(time.time())

        # Save MP4
        mp4_path = output_dir / f"{input_name}_svd_{world_size}gpu_{timestamp}.mp4"
        save_video(video_frames, str(mp4_path), args.fps)

        # Save GIF
        gif_path = output_dir / f"{input_name}_svd_{world_size}gpu_{timestamp}.gif"
        save_gif(video_frames, str(gif_path), args.fps)

        # Save input image for comparison
        input_copy_path = output_dir / f"{input_name}_input_{timestamp}.png"
        image.save(str(input_copy_path))
        LOGGER.info(f"Input image saved to: {input_copy_path}")

        # Print summary
        total_time = load_time + diffusion_time + decode_time
        LOGGER.info("=" * 50)
        LOGGER.info("Generation Summary")
        LOGGER.info("=" * 50)
        LOGGER.info(f"  Input: {args.input_image}")
        LOGGER.info(f"  Output: {mp4_path}")
        LOGGER.info(f"  GPUs: {world_size}")
        LOGGER.info(f"  Steps: {args.total_steps}")
        LOGGER.info(f"  Frames: {args.num_frames}")
        LOGGER.info(f"  Resolution: {args.width}x{args.height}")
        LOGGER.info(f"  Model load time: {load_time:.2f}s")
        LOGGER.info(f"  Diffusion time: {diffusion_time:.2f}s")
        LOGGER.info(f"  Decode time: {decode_time:.2f}s")
        LOGGER.info(f"  Total time: {total_time:.2f}s")
        LOGGER.info("=" * 50)

    # Cleanup
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
