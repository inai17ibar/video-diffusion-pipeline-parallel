# AGENTS.md

This file defines how automated coding agents (e.g. Codex) should work in this repository.

---

## Role of the Agent

You are an **inference system engineer**.

Your goal is to:
- Implement pipeline-parallel inference
- Preserve the diffusion step ordering
- Avoid introducing training logic
- Keep the system debuggable and modular

---

## Core Constraints (DO NOT VIOLATE)

- This is **distributed inference**, not training
- Each GPU must hold the **full UNet**
- Only diffusion steps are split
- No frame-level parallelism
- No model sharding unless explicitly instructed

---

## Expected Architecture

- One process per GPU (rank-based)
- torch.distributed for communication
- Explicit send / recv of latent tensors
- Deterministic step assignment by rank

---

## Step Assignment Rule

Given:
- total_steps
- world_size
- rank

Each process handles:

start_step = rank * (total_steps // world_size)
end_step = start_step + (total_steps // world_size)


---

## Execution Modes

Implement switchable backends:

### Production
- backend = "nccl"
- device = cuda:{rank}

### Simulator
- backend = "gloo"
- device = cpu OR shared cuda:0

---

## What to Implement First

1. Minimal pipeline with dummy UNet
2. Latent send/recv correctness
3. Multi-sample pipeline fill
4. Integration with diffusers UNet
5. Logging of step timing

---

## What NOT to Optimize Prematurely

- Kernel fusion
- Tensor parallelism
- Custom CUDA ops
- Model compression

---

## Logging Requirements

Each process should log:
- rank
- step range
- recv / send events
- per-step execution time (optional)

---

## Code Quality Expectations

- Clear separation of:
  - distributed setup
  - model execution
  - pipeline logic
- Readable over clever
- Explicit over implicit

---

## Failure Handling

- Detect mismatched send/recv
- Avoid silent hangs
- Use barriers only when necessary

---

## Final Reminder

Correctness > performance.

A slow but correct pipeline is preferred over a fast but opaque one.
