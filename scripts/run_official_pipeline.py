#!/usr/bin/env python3
"""Run official diffusers SVD pipeline with CPU offloading for baseline comparison."""

import imageio
import numpy as np
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image

print("Loading pipeline...")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_sequential_cpu_offload()

# Load and resize image
image = Image.open("demo_input_photo.jpg").convert("RGB").resize((1024, 576))

print("Generating video (25 steps, CFG 1.0-3.0)...")
generator = torch.manual_seed(42)
frames = pipe(
    image,
    num_frames=14,
    num_inference_steps=25,
    decode_chunk_size=4,
    min_guidance_scale=1.0,
    max_guidance_scale=3.0,
    fps=7,
    motion_bucket_id=127,
    noise_aug_strength=0.02,
    generator=generator,
).frames[0]

frames_np = [np.array(f) for f in frames]
imageio.mimsave("outputs/official_full_baseline.mp4", frames_np, fps=7)
imageio.mimsave("outputs/official_full_baseline.gif", frames_np, fps=7, loop=0)
arr = np.array(frames_np)
print(f"Saved {len(frames_np)} frames")
print(f"Stats: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.1f}")

# Save individual frames
for i in [0, 3, 7, 13]:
    Image.fromarray(frames_np[i]).save(f"outputs/official_baseline_frame{i}.png")
    print(f"Frame {i}: mean={frames_np[i].mean():.1f}")
