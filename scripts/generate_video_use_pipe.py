#!/usr/bin/env python3
"""Generate video using official pipeline components for encoding/decoding.

Uses the exact same encode/decode as the official StableVideoDiffusionPipeline
but with our multi-GPU denoising loop.
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

    if is_distributed:
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # Load the FULL official pipeline to use its exact encoding/decoding
    LOGGER.info(f"Rank {rank}: Loading official pipeline...")
    start_load = time.time()

    from diffusers import StableVideoDiffusionPipeline

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)

    load_time = time.time() - start_load
    LOGGER.info(f"Rank {rank}: Pipeline loaded in {load_time:.2f}s")

    # Load image using our center crop
    from PIL import Image

    image_pil = Image.open(args.input_image).convert("RGB")
    src_w, src_h = image_pil.size
    scale = max(args.width / src_w, args.height / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    if (new_w, new_h) != (src_w, src_h):
        image_pil = image_pil.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - args.width) // 2
    top = (new_h - args.height) // 2
    image_pil = image_pil.crop((left, top, left + args.width, top + args.height))

    # === Use official pipeline's exact encoding ===
    do_cfg = args.max_guidance_scale > 1.0

    # CLIP encoding (official method)
    image_embeddings = pipe._encode_image(image_pil, device, 1, do_cfg)
    LOGGER.info(
        f"Rank {rank}: CLIP embeddings: {image_embeddings.shape}, mean={image_embeddings.mean():.4f}"
    )

    # VAE encoding (official method with force_upcast)
    image_for_vae = pipe.video_processor.preprocess(
        image_pil, height=args.height, width=args.width
    ).to(device)
    from diffusers.utils.torch_utils import randn_tensor

    torch.manual_seed(args.seed)
    noise = randn_tensor(
        image_for_vae.shape,
        generator=torch.Generator(device).manual_seed(args.seed),
        device=device,
        dtype=image_for_vae.dtype,
    )
    image_for_vae = image_for_vae + args.noise_aug_strength * noise

    needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
    if needs_upcasting:
        pipe.vae.to(dtype=torch.float32)

    image_latents = pipe._encode_vae_image(image_for_vae, device, 1, do_cfg)
    image_latents = image_latents.to(image_embeddings.dtype)

    if needs_upcasting:
        pipe.vae.to(dtype=torch.float16)

    # Repeat for frames
    image_latents = image_latents.unsqueeze(1).repeat(1, args.num_frames, 1, 1, 1)
    LOGGER.info(
        f"Rank {rank}: image_latents: {image_latents.shape}, mean={image_latents.mean():.4f}"
    )

    # added_time_ids (official method)
    added_time_ids = pipe._get_add_time_ids(
        args.fps - 1,
        args.motion_bucket_id,
        args.noise_aug_strength,
        image_embeddings.dtype,
        1,
        1,
        do_cfg,
    ).to(device)

    # Free encoder to save memory
    del pipe.image_encoder, pipe.feature_extractor
    if rank != world_size - 1:
        # Non-final ranks don't need VAE for decoding
        del pipe.vae
    torch.cuda.empty_cache()

    # === Scheduler setup ===
    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.total_steps, device=device)

    # === Prepare latents (official method) ===
    generator = torch.Generator(device).manual_seed(args.seed)
    latents = pipe.prepare_latents(
        1,
        args.num_frames,
        pipe.unet.config.in_channels,
        args.height,
        args.width,
        image_embeddings.dtype,
        device,
        generator,
        None,
    )
    LOGGER.info(
        f"Rank {rank}: latents: {latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}"
    )

    # Guidance scale
    guidance_scale = torch.linspace(
        args.min_guidance_scale, args.max_guidance_scale, args.num_frames
    )
    guidance_scale = guidance_scale.unsqueeze(0).to(device, latents.dtype)
    guidance_scale = guidance_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # === DENOISING LOOP ===
    LOGGER.info(f"Rank {rank}: Running {args.total_steps} steps...")
    start_diffusion = time.time()
    unet = pipe.unet

    if is_distributed:
        from src.pipeline.step_assignment import assign_steps

        step_range = assign_steps(args.total_steps, world_size, rank)
        LOGGER.info(f"Rank {rank}: steps {step_range.start}-{step_range.end - 1}")

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
                    latent_model_input,
                    t,
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
            LOGGER.info(f"Rank {rank}: step {i} in {(time.time()-step_start)*1000:.0f}ms")

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
                    latent_model_input,
                    t,
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
            if i % 5 == 0 or i == args.total_steps - 1:
                LOGGER.info(f"Step {i}/{args.total_steps} in {(time.time()-step_start)*1000:.0f}ms")

    diffusion_time = time.time() - start_diffusion

    # === DECODE (official method) ===
    if rank == world_size - 1:
        LOGGER.info("Decoding (official method)...")
        # Free UNet to make room for VAE decode
        del unet
        del pipe.unet
        del image_embeddings, image_latents, added_time_ids
        torch.cuda.empty_cache()

        # Move VAE to CPU for decoding to avoid OOM
        pipe.vae.to("cpu", dtype=torch.float32)
        latents_cpu = latents.to("cpu", dtype=torch.float32)
        torch.cuda.empty_cache()

        # Manual decode (same as pipe.decode_latents but on CPU)
        latents_flat = latents_cpu.flatten(0, 1)  # (B*F, C, H, W)
        latents_flat = latents_flat / pipe.vae.config.scaling_factor
        frames_list = []
        with torch.no_grad():
            for i in range(0, latents_flat.shape[0], args.decode_chunk_size):
                chunk = latents_flat[i : i + args.decode_chunk_size]
                frame = pipe.vae.decode(chunk, num_frames=chunk.shape[0]).sample
                frames_list.append(frame)
        frames = torch.cat(frames_list, dim=0)
        frames = frames.reshape(1, args.num_frames, *frames.shape[1:])

        # Postprocess: [-1, 1] -> [0, 255] PIL images
        frames_np_list = []
        for f_idx in range(args.num_frames):
            frame = frames[0, f_idx]  # (3, H, W)
            frame = ((frame + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            frame = frame.permute(1, 2, 0).numpy()  # (H, W, 3)
            frames_np_list.append(frame)

        import imageio
        import numpy as np

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        name = Path(args.input_image).stem

        mp4_path = output_dir / f"{name}_pipe_{world_size}gpu_{ts}.mp4"
        gif_path = output_dir / f"{name}_pipe_{world_size}gpu_{ts}.gif"

        imageio.mimsave(str(mp4_path), frames_np_list, fps=args.fps)
        imageio.mimsave(str(gif_path), frames_np_list, fps=args.fps, loop=0)
        image_pil.save(str(output_dir / f"{name}_input_{ts}.png"))

        arr = np.array(frames_np_list)
        total_time = load_time + diffusion_time
        LOGGER.info("=" * 50)
        LOGGER.info(f"  Output: {mp4_path}")
        LOGGER.info(f"  GPUs: {world_size}, Steps: {args.total_steps}, Frames: {args.num_frames}")
        LOGGER.info(f"  Diffusion: {diffusion_time:.2f}s, Total: {total_time:.2f}s")
        LOGGER.info(
            f"  Stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.1f}"
        )
        LOGGER.info("=" * 50)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
