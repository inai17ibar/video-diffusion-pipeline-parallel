# Context and Design Rationale

This document explains *why* the system is designed this way.
It exists to prevent future contributors (and Codex) from making incorrect assumptions.

---

## Problem Statement

Video diffusion models are slow because:

- Diffusion steps are **strictly sequential**
- Each step requires a heavy UNet forward
- Frame-wise parallelism breaks temporal consistency
- Model parallelism adds communication overhead

---

## Why Pipeline Parallelism?

We parallelize **over time (diffusion steps)**, not:

- ❌ frames (breaks motion consistency)
- ❌ model layers (communication-heavy)
- ❌ data parallel (ineffective for inference latency)

Pipeline parallelism allows:

- Structural use of multiple GPUs
- High utilization in steady state
- Clear reasoning about correctness

---

## What Is Being Parallelized?

- **Only diffusion steps**
- Not frames
- Not model parameters

Each GPU:
- Holds a full copy of the UNet
- Executes only a subset of diffusion steps
- Passes latent tensors downstream

---

## Why 7 GPUs?

- Diffusion steps (25–35) split evenly
- Enough stages to demonstrate pipeline behavior
- Realistic scale for enterprise inference nodes

This design generalizes to N GPUs.

---

## Why Not Faster for a Single Video?

Pipeline parallelism reduces **idle time**, not initial latency.

- First sample pays pipeline fill cost
- Subsequent samples benefit from overlap

This matches real-world workloads:
- Retry generation
- Multi-candidate generation
- Batch user requests

---

## VRAM Considerations

Per GPU usage (fp16 inference):

- UNet + activations: ~6–8 GB
- Text encoder: ~1 GB
- Buffers + overhead: ~4–6 GB
- **Total: ~14–18 GB**

Fits comfortably within A5000 (24 GB).

---

## Simulator Philosophy

We distinguish between:

- **Structural correctness** (can be simulated)
- **Performance correctness** (requires real GPUs)

Simulator modes are intentionally supported to:
- Validate step assignment
- Debug deadlocks
- Verify pipeline flow

---

## Non-Goals

This project does NOT aim to:

- Train video diffusion models
- Implement large-scale multi-node inference
- Achieve SOTA generation quality

---

## Design Principle

> GPUs should be used to **reduce decision latency**, not just raw compute time.

