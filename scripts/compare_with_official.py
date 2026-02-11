#!/usr/bin/env python3
"""Compare our SVD wrapper with the official diffusers pipeline step by step."""

import torch
from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from diffusers.models import UNetSpatioTemporalConditionModel
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

DEVICE = torch.device("cuda:0")
DTYPE = torch.float16
MODEL_ID = "stabilityai/stable-video-diffusion-img2vid-xt"
NUM_FRAMES = 14
NUM_STEPS = 21
HEIGHT, WIDTH = 576, 1024
SEED = 42


def main():
    # Load models
    print("Loading models...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_ID, subfolder="image_encoder", torch_dtype=DTYPE
    ).to(DEVICE)
    image_encoder.eval()
    feature_extractor = CLIPImageProcessor.from_pretrained(MODEL_ID, subfolder="feature_extractor")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=DTYPE
    ).to(DEVICE)
    vae.eval()
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=DTYPE
    ).to(DEVICE)
    unet.eval()

    # Load image
    image = Image.open("demo_input_photo.jpg").convert("RGB")
    image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)

    # === OFFICIAL PIPELINE ENCODING ===
    print("\n=== OFFICIAL ENCODING ===")

    # CLIP encoding
    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        clip_embeds = image_encoder(pixel_values).image_embeds
        clip_embeds = clip_embeds.unsqueeze(1)
    print(f"CLIP embeddings: {clip_embeds.shape}, mean={clip_embeds.mean():.4f}")

    # VAE encoding (official way: normalize to [-1,1], noise in pixel space, .mode(), NO scaling_factor)
    from torchvision import transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    img_tensor = transform(image).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Add noise in pixel space
    torch.manual_seed(SEED)
    noise = torch.randn_like(img_tensor)
    img_tensor_noised = img_tensor + 0.02 * noise

    with torch.no_grad():
        image_latents = vae.encode(img_tensor_noised).latent_dist.mode()
    print(
        f"Image latents (NO scaling): {image_latents.shape}, mean={image_latents.mean():.4f}, std={image_latents.std():.4f}"
    )

    # Repeat for frames: (B, C, H, W) -> (B, F, C, H, W) (official convention)
    image_latents_official = image_latents.unsqueeze(1).repeat(1, NUM_FRAMES, 1, 1, 1)
    print(f"Image latents repeated: {image_latents_official.shape}")

    # Setup scheduler
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
    scheduler.set_timesteps(NUM_STEPS, device=DEVICE)
    print(f"Scheduler timesteps[0]: {scheduler.timesteps[0]}")
    print(f"Scheduler sigmas[0]: {scheduler.sigmas[0]:.4f}")
    print(f"init_noise_sigma: {scheduler.init_noise_sigma:.4f}")

    # Prepare latents (official convention: B, F, C, H, W)
    latent_h, latent_w = HEIGHT // 8, WIDTH // 8
    torch.manual_seed(SEED)
    latents = torch.randn(1, NUM_FRAMES, 4, latent_h, latent_w, device=DEVICE, dtype=DTYPE)
    latents = latents * scheduler.init_noise_sigma
    print(f"Initial latents: {latents.shape}, mean={latents.mean():.4f}, std={latents.std():.4f}")

    # added_time_ids
    added_time_ids = torch.tensor(
        [[6, 127, 0.02]], dtype=DTYPE, device=DEVICE
    )  # fps-1, motion_bucket, noise_aug

    # === RUN ONE OFFICIAL STEP ===
    print("\n=== OFFICIAL STEP 0 ===")
    t = scheduler.timesteps[0]
    latent_model_input = scheduler.scale_model_input(latents, t)
    print(
        f"After scale_model_input: mean={latent_model_input.mean():.4f}, std={latent_model_input.std():.4f}"
    )

    # Concat image latents (dim=2 in B,F,C,H,W convention)
    latent_model_input = torch.cat([latent_model_input, image_latents_official], dim=2)
    print(f"After concat: {latent_model_input.shape}, mean={latent_model_input.mean():.4f}")

    with torch.no_grad():
        noise_pred_official = unet(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=clip_embeds,
            added_time_ids=added_time_ids,
            return_dict=False,
        )[0]
    print(
        f"UNet output: {noise_pred_official.shape}, mean={noise_pred_official.mean():.4f}, std={noise_pred_official.std():.4f}"
    )

    # Scheduler step
    result_official = scheduler.step(noise_pred_official, t, latents).prev_sample
    print(f"After step: mean={result_official.mean():.4f}, std={result_official.std():.4f}")

    # === RUN OUR WRAPPER STEP 0 ===
    print("\n=== OUR WRAPPER STEP 0 ===")
    from src.models.svd_unet import StableVideoUNet

    timesteps_list = StableVideoUNet._default_timestep_schedule(NUM_STEPS)
    model = StableVideoUNet(unet=unet, timesteps=timesteps_list, dtype=DTYPE)

    # Set conditioning
    image_latents_ours = image_latents.unsqueeze(2).repeat(1, 1, NUM_FRAMES, 1, 1)  # (B,C,F,H,W)
    model.set_conditioning(
        image_embeddings=clip_embeds,
        image_latents=image_latents_ours,
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        guidance_scale=None,
        num_frames=NUM_FRAMES,
    )

    # Prepare our latents (B, C, F, H, W)
    torch.manual_seed(SEED)
    latents_ours = torch.randn(1, 4, NUM_FRAMES, latent_h, latent_w, device=DEVICE, dtype=DTYPE)
    latents_ours = latents_ours * model.init_noise_sigma
    print(
        f"Our latents: {latents_ours.shape}, mean={latents_ours.mean():.4f}, std={latents_ours.std():.4f}"
    )

    # Compare: our (B,C,F,H,W) permuted should match official (B,F,C,H,W)
    latents_ours_bfchw = latents_ours.permute(0, 2, 1, 3, 4)
    diff = (latents_ours_bfchw - latents).abs().mean()
    print(f"Latent diff (ours permuted vs official): {diff:.6f}")

    # Run step
    result_ours = model(latents_ours, step=0)
    print(
        f"Our result: {result_ours.shape}, mean={result_ours.mean():.4f}, std={result_ours.std():.4f}"
    )

    # Compare results
    result_ours_bfchw = result_ours.permute(0, 2, 1, 3, 4)
    diff = (result_ours_bfchw - result_official).abs()
    print("\n=== COMPARISON ===")
    print(f"Step 0 result diff: mean={diff.mean():.6f}, max={diff.max():.6f}")

    # Decode and compare
    print("\n=== DECODING ===")
    # Official decode
    decode_latents = result_official.clone()
    for _ in range(1, NUM_STEPS):
        t = scheduler.timesteps[_]
        lmi = scheduler.scale_model_input(decode_latents, t)
        lmi = torch.cat([lmi, image_latents_official], dim=2)
        with torch.no_grad():
            np_ = unet(
                sample=lmi,
                timestep=t,
                encoder_hidden_states=clip_embeds,
                added_time_ids=added_time_ids,
                return_dict=False,
            )[0]
        decode_latents = scheduler.step(np_, t, decode_latents).prev_sample

    print(f"Final latents: mean={decode_latents.mean():.4f}, std={decode_latents.std():.4f}")

    # Decode video
    dec_input = decode_latents.flatten(0, 1)  # (B*F, C, H, W)
    dec_input = dec_input / vae.config.scaling_factor
    with torch.no_grad():
        frames = vae.decode(dec_input, num_frames=NUM_FRAMES).sample
    print(f"Decoded frames: {frames.shape}, mean={frames.mean():.4f}, std={frames.std():.4f}")

    # Save
    import imageio

    frames = frames.reshape(1, NUM_FRAMES, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
    frames_np = frames[0].permute(1, 2, 3, 0)
    frames_np = ((frames_np.float() + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    imageio.mimsave("outputs/official_step_compare.mp4", frames_np, fps=7)
    print(
        f"Saved official decode: min={frames_np.min()}, max={frames_np.max()}, mean={frames_np.mean():.1f}"
    )


if __name__ == "__main__":
    main()
