# video-diffusion-pipeline-parallel

Pipeline parallel inference for video diffusion models by splitting **diffusion steps (temporal axis)** across multiple GPUs.

This repository explores how **A5000 × 7 GPUs** can be used to accelerate video generation not by model or frame parallelism, but by **diffusion step pipeline parallelism**.

---

## Motivation

Video diffusion models are slow because diffusion steps must be executed **sequentially**.

Increasing GPU count alone does not reduce latency unless we restructure computation.
This project addresses the bottleneck by **pipelining diffusion steps across GPUs**, enabling higher throughput in multi-sample generation scenarios.

---

## Key Idea

- Each GPU holds the **same UNet model**
- Diffusion steps are split across GPUs
- Latent tensors are passed GPU-to-GPU in step order
- This is **distributed inference**, not distributed training

Noise
↓ GPU0 (step 0–4)
↓ GPU1 (step 5–9)
↓ GPU2 (step 10–14)
↓ GPU3 (step 15–19)
↓ GPU4 (step 20–24)
↓ GPU5 (step 25–29)
↓ GPU6 (step 30–34)
→ Video


---

## Target Model

- Stable Video Diffusion (SVD) or similar UNet-based video diffusion models
- fp16 inference
- 14–25 frames, 25–35 diffusion steps

---

## Expected Performance (Reference)

| Configuration | Time per video |
|--------------|----------------|
| Single GPU | 40–60 sec |
| 7 GPU (first sample) | ~35 sec |
| 7 GPU (steady state) | 8–12 sec |

> Throughput improves significantly when generating multiple videos sequentially.

---

## Hardware / Environment

- Linux (Ubuntu 20.04 / 22.04)
- NVIDIA GPU × 7 (A5000 recommended)
- CUDA 11.8+
- PyTorch 2.x
- torch.distributed (NCCL)

---

## Development Modes

- **Real mode**: 1 node, 7 GPUs, NCCL
- **Simulator mode**:
  - CPU + Gloo backend
  - 1 GPU shared by multiple processes
  - Used for logic verification (not performance)

---

## Project Status

- [x] Pipeline design
- [x] Step-split prototype
- [ ] Multi-sample pipeline fill
- [ ] diffusers integration
- [ ] Performance profiling

---

## Disclaimer

This project focuses on **system design and inference structure**.
It is not intended to compete with end-to-end optimized production systems.

---

## License

MIT
