#!/usr/bin/env python3
"""Generate video using official diffusers scheduler.step() directly.

This script bypasses our StableVideoUNet wrapper and uses the exact same
denoising loop as the official SVD pipeline, to isolate whether the issue
is in our Euler step implementation.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--total-steps", type=int, default=25)
    parser.add_argument("--num-frames", type=int, default=14)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--min-guidance-scale", type=float, default=1.0)
    parser.add_argument("--max-guidance-scale", type=float, default=3.0)
    parser.add_argument("--motion-bucket-id", type=int, default=127)
    parser.add_argument("--noise-aug-strength", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-id", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt"
    )
    parser.add_argument("--decode-chunk-size", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    if is_distributed:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Load models
    LOGGER.info(f"Rank {rank}: Loading models...")
    start_load = time.time()

    from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
    from diffusers.models import UNetSpatioTemporalConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

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

    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.model_id, subfolder="unet", torch_dtype=dtype
    ).to(device)
    unet.eval()

    # Memory optimization
    import contextlib

    with contextlib.suppress(Exception):
        unet.enable_xformers_memory_efficient_attention()

    load_time = time.time() - start_load
    LOGGER.info(f"Rank {rank}: Models loaded in {load_time:.2f}s")

    # === ENCODING (exactly matching official pipeline) ===
    from PIL import Image
    from torchvision import transforms

    image = Image.open(args.input_image).convert("RGB")
    # Center crop (same as our demo)
    src_w, src_h = image.size
    scale = max(args.width / src_w, args.height / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    if (new_w, new_h) != (src_w, src_h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - args.width) // 2
    top = (new_h - args.height) // 2
    image = image.crop((left, top, left + args.width, top + args.height))

    # CLIP encoding
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device, dtype=dtype)
    with torch.no_grad():
        image_embeddings = image_encoder(pixel_values).image_embeds.unsqueeze(1)  # (1, 1, 1024)

    # VAE encoding (official convention: noise in pixel space, .mode(), NO scaling_factor)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    img_tensor = transform(image).unsqueeze(0).to(device, dtype=dtype)

    torch.manual_seed(args.seed)
    noise = torch.randn_like(img_tensor)
    img_tensor = img_tensor + args.noise_aug_strength * noise

    with torch.no_grad():
        image_latents = vae.encode(img_tensor).latent_dist.mode()  # (1, 4, H/8, W/8)

    # CFG: duplicate with zeros for unconditional
    do_cfg = args.max_guidance_scale > 1.0
    if do_cfg:
        image_latents = torch.cat(
            [torch.zeros_like(image_latents), image_latents]
        )  # (2, 4, H/8, W/8)
        image_embeddings = torch.cat(
            [torch.zeros_like(image_embeddings), image_embeddings]
        )  # (2, 1, 1024)

    # Repeat for frames: (B, C, H, W) -> (B, F, C, H, W)
    image_latents = image_latents.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)

    # added_time_ids (official: fps-1)
    added_time_ids = torch.tensor(
        [[args.fps - 1, args.motion_bucket_id, args.noise_aug_strength]], dtype=dtype, device=device
    )
    if do_cfg:
        added_time_ids = added_time_ids.repeat(2, 1)

    # Free encoder
    del image_encoder, feature_extractor
    torch.cuda.empty_cache()

    # === SCHEDULER SETUP ===
    scheduler = EulerDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="v_prediction",
        interpolation_type="linear",
        timestep_spacing="leading",
        steps_offset=1,
        use_karras_sigmas=True,
    )
    scheduler.set_timesteps(args.total_steps, device=device)

    # === PREPARE LATENTS (official convention: B, F, C, H, W) ===
    latent_h, latent_w = args.height // 8, args.width // 8
    torch.manual_seed(args.seed)
    latents = torch.randn(1, args.num_frames, 4, latent_h, latent_w, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    # Per-frame guidance scale
    guidance_scale = torch.linspace(
        args.min_guidance_scale, args.max_guidance_scale, args.num_frames
    )
    guidance_scale = guidance_scale.unsqueeze(0).to(device, dtype)  # (1, F)
    # Expand to (1, F, 1, 1, 1) for broadcasting with (B, F, C, H, W)
    guidance_scale = guidance_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    LOGGER.info(f"Rank {rank}: init_noise_sigma={scheduler.init_noise_sigma:.4f}")
    LOGGER.info(f"Rank {rank}: latents mean={latents.mean():.4f}, std={latents.std():.4f}")
    LOGGER.info(
        f"Rank {rank}: image_latents mean={image_latents.mean():.4f}, std={image_latents.std():.4f}"
    )

    # === DENOISING LOOP (exactly matching official pipeline) ===
    LOGGER.info(f"Rank {rank}: Running {args.total_steps} steps...")
    start_diffusion = time.time()

    if is_distributed:
        from src.pipeline.step_assignment import assign_steps

        step_range = assign_steps(args.total_steps, world_size, rank)
        LOGGER.info(f"Rank {rank}: processing steps {step_range.start} to {step_range.end - 1}")

        if rank > 0:
            latents = torch.empty_like(latents)
            dist.recv(latents, src=rank - 1)

        for i in range(step_range.start, step_range.end):
            t = scheduler.timesteps[i]
            step_start = time.time()

            with torch.inference_mode():
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                noise_pred = unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()
            step_time = (time.time() - step_start) * 1000
            LOGGER.info(f"Rank {rank}: step {i} completed in {step_time:.2f}ms")

        if rank < world_size - 1:
            dist.send(latents, dst=rank + 1)
    else:
        for i, t in enumerate(scheduler.timesteps):
            step_start = time.time()
            with torch.inference_mode():
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                noise_pred = unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=image_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                if do_cfg:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

            latents = scheduler.step(noise_pred, t, latents).prev_sample
            torch.cuda.empty_cache()
            step_time = (time.time() - step_start) * 1000
            if i % 5 == 0 or i == args.total_steps - 1:
                LOGGER.info(f"Step {i}/{args.total_steps} in {step_time:.2f}ms")

    diffusion_time = time.time() - start_diffusion

    # === DECODE (official convention) ===
    if rank == world_size - 1:
        LOGGER.info("Decoding...")
        start_decode = time.time()

        dec_latents = latents.flatten(0, 1)  # (B*F, C, H, W)
        dec_latents = dec_latents / vae.config.scaling_factor

        frames = []
        with torch.no_grad():
            for i in range(0, dec_latents.shape[0], args.decode_chunk_size):
                chunk = dec_latents[i : i + args.decode_chunk_size]
                frame = vae.decode(chunk, num_frames=chunk.shape[0]).sample
                frames.append(frame)
        frames = torch.cat(frames, dim=0)  # (B*F, 3, H, W)
        frames = frames.reshape(1, args.num_frames, *frames.shape[1:])  # (1, F, 3, H, W)

        decode_time = time.time() - start_decode

        # Save
        import imageio

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames_np = frames[0].permute(0, 2, 3, 1)  # (F, H, W, 3)
        frames_np = ((frames_np.float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        ts = int(time.time())
        name = Path(args.input_image).stem
        mp4_path = output_dir / f"{name}_official_loop_{world_size}gpu_{ts}.mp4"
        gif_path = output_dir / f"{name}_official_loop_{world_size}gpu_{ts}.gif"

        imageio.mimsave(str(mp4_path), frames_np, fps=args.fps)
        imageio.mimsave(str(gif_path), frames_np, fps=args.fps, loop=0)
        image.save(str(output_dir / f"{name}_input_{ts}.png"))

        total_time = load_time + diffusion_time + decode_time
        LOGGER.info("=" * 50)
        LOGGER.info(f"  Output: {mp4_path}")
        LOGGER.info(f"  GPUs: {world_size}, Steps: {args.total_steps}")
        LOGGER.info(f"  Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")
        LOGGER.info(
            f"  Diffusion: {diffusion_time:.2f}s, Decode: {decode_time:.2f}s, Total: {total_time:.2f}s"
        )
        arr = frames_np
        LOGGER.info(
            f"  Stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.1f}"
        )
        LOGGER.info("=" * 50)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
