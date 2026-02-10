# CLAUDE.md

This file provides guidance for AI assistants working in this repository.

## Project Overview

Pipeline-parallel inference for video diffusion models. The core idea: distribute diffusion steps (temporal axis) across multiple GPUs, where each GPU holds the **complete UNet model** and processes a contiguous subset of denoising steps. This is **inference only** — no training logic.

- **Target model**: Stable Video Diffusion (SVD) — 14-25 frames, 25-35 diffusion steps
- **Default topology**: 7 GPUs, each running a slice of the diffusion step sequence
- **Communication**: `torch.distributed` point-to-point `send`/`recv` of latent tensors between ranks

## Repository Structure

```
src/
  pipeline/
    pipeline.py          # Core engine: PipelineStage, PipelineConfig, LatentSpec
    step_assignment.py   # Deterministic step range assignment per rank
  models/
    dummy_unet.py        # Lightweight synthetic UNet for testing (Conv3d-based)
    svd_unet.py          # Real SVD UNet wrapper around diffusers
  modes/
    simulator.py         # Entry point: CPU/Gloo testing mode
    production.py        # Entry point: multi-GPU NCCL production mode
  distributed/
    setup.py             # torch.distributed init/finalize
    backend.py           # Backend resolution (nccl vs gloo)
tests/
  test_step_assignment.py  # Step distribution logic tests
  test_dummy_unet.py       # Model shape/behavior tests
docs/
  context.md             # Design rationale
  benchmark.md           # Benchmark guide (Japanese)
```

## Common Commands

### Install dependencies

```bash
pip install -r requirements-dev.txt
```

### Run tests

```bash
pytest tests/ -v
```

Tests run with `addopts = "-v --tb=short"` by default (configured in `pyproject.toml`).

### Lint and format

```bash
ruff check src/ tests/            # Lint
ruff check src/ tests/ --fix      # Lint with auto-fix
ruff format src/ tests/           # Format
```

### Run simulator mode (no GPU required)

```bash
torchrun --nproc_per_node=4 -m src.modes.simulator \
  --total-steps 28 --device cpu --dtype fp32
```

### Run production mode (multi-GPU)

```bash
torchrun --nproc_per_node=7 -m src.modes.production \
  --total-steps 28 --latent-shape 1 4 14 64 64 \
  --num-samples 10 --enable-memory-opt
```

## Pre-commit Hooks

Pre-commit is configured (`.pre-commit-config.yaml`) and runs on every commit:

1. Trailing whitespace removal, EOF fixer, YAML validation, large file check, merge conflict detection
2. **Ruff lint** with `--fix --exit-non-zero-on-fix`
3. **Ruff format**
4. **pytest** (`pytest tests/ -x -q`) — tests must pass before commit

The pytest hook activates the virtualenv at `venv/bin/activate`.

## Code Style

- **Formatter/Linter**: Ruff (configured in `pyproject.toml`)
- **Python**: >= 3.9
- **Line length**: 100
- **Quote style**: double quotes
- **Indent style**: spaces
- **Import sorting**: isort via Ruff, `src` is first-party
- **Lint rules**: E, W, F, I, B, C4, UP, ARG, SIM (with E501, B008, B905 ignored)

## Architecture Constraints

These constraints are critical and must not be violated:

1. **Inference only** — never introduce training logic (optimizers, loss, backward passes)
2. **Full UNet per GPU** — every rank holds the complete model; no model sharding or tensor parallelism
3. **Step-level split only** — diffusion steps are divided across ranks; no frame-level parallelism
4. **Deterministic step assignment** — rank N handles steps `[N * (total_steps // world_size), (N+1) * (total_steps // world_size))`. `total_steps` must be evenly divisible by `world_size`
5. **Explicit send/recv** — latent tensors are passed between ranks via `torch.distributed.send()` / `torch.distributed.recv()`, not collective operations
6. **Correctness over performance** — a slow but correct pipeline is preferred over a fast but opaque one

## Key Design Patterns

- **`PipelineStage`** (in `src/pipeline/pipeline.py`): manages a single rank's execution loop — receives latent from previous rank, runs local steps, sends to next rank
- **`StepRange`** (in `src/pipeline/step_assignment.py`): dataclass representing `[start, end)` step interval
- **Model interface**: all models implement `forward(latent, step) -> latent`
- **Two execution modes**: simulator (Gloo/CPU, DummyUNet) and production (NCCL/CUDA, real SVD UNet)

## Environment Variables

Set automatically by `torchrun`:
- `RANK` — global process rank
- `LOCAL_RANK` — local GPU index
- `WORLD_SIZE` — total number of processes

Optional:
- `PIPELINE_BACKEND` — override backend selection (`nccl` or `gloo`)

## What Not to Do

- Do not add kernel fusion, tensor parallelism, custom CUDA ops, or model compression unless explicitly instructed
- Do not use collective operations (all_reduce, broadcast) for latent passing — use point-to-point send/recv
- Do not add training-related code
- Do not shard the model across GPUs
- Avoid silent hangs — detect mismatched send/recv and use barriers only when necessary
