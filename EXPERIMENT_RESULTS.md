# Experimental Results: Pipeline Parallel Video Diffusion

**Date**: 2026-02-08
**Environment**: Ubuntu 22.04, 7x NVIDIA RTX A5000 (24GB each)
**PyTorch Version**: 2.10.0+cu128

---

## 1. Environment Setup

### Hardware Configuration
| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A5000 x 7 |
| GPU Memory | 24564 MiB per GPU |
| Platform | Linux 5.15.0-168-generic |

### Software Dependencies
```
torch==2.10.0
diffusers==0.36.0
transformers==5.1.0
accelerate==1.12.0
```

---

## 2. Simulator Mode Experiments (DummyUNet)

The simulator mode uses a lightweight DummyUNet to verify pipeline parallelism logic without requiring the full SVD model.

### Single GPU Baseline (NCCL)
```
Device: cuda:0
Total Steps: 28
First step: 162.34 ms (includes CUDA initialization)
Subsequent steps: ~0.3-0.5 ms
Total time: 3.99 seconds
```

### Multi-Process Tests (Gloo/CPU)

| Processes | Steps per Process | Total Time | Pipeline Verification |
|-----------|-------------------|------------|----------------------|
| 1 | 28 | 3.99 sec | Baseline |
| 2 | 14 | 4.22 sec | Pass |
| 4 | 7 | 4.15 sec | Pass |
| 7 | 4 | 4.43 sec | Pass |

**Observations**:
- Pipeline handoff between ranks works correctly
- Each rank correctly receives latent from previous rank
- Final rank produces output with expected tensor norm
- Communication overhead visible in multi-process setup

### Step Distribution (7 processes, 28 total steps)
```
Rank 0: steps 27, 26, 25, 24 -> sends to Rank 1
Rank 1: steps 23, 22, 21, 20 -> sends to Rank 2
Rank 2: steps 19, 18, 17, 16 -> sends to Rank 3
Rank 3: steps 15, 14, 13, 12 -> sends to Rank 4
Rank 4: steps 11, 10, 9, 8   -> sends to Rank 5
Rank 5: steps 7, 6, 5, 4     -> sends to Rank 6
Rank 6: steps 3, 2, 1, 0     -> outputs final result
```

---

## 3. Production Mode Issues

### Issue 1: Latent Channel Mismatch
**Error**: `RuntimeError: expected input to have 8 channels, but got 4 channels`

**Root Cause**:
- The SVD UNet expects 8-channel input (noisy latent + image conditioning)
- The UNet outputs 4-channel noise prediction
- The pipeline passes UNet output directly as next step input

**Location**: `src/models/svd_unet.py:217`

### Issue 2: GPU Memory (OOM)
**Error**: `torch.OutOfMemoryError: CUDA out of memory`

**Configuration**: `--latent-shape 1 8 14 64 64`

**Details**:
- Model size: ~10GB
- Peak memory usage exceeded 24GB GPU memory
- Occurs during UNet forward pass with full resolution

### Issue 3: Multi-GPU Device Assignment
**Error**: `Duplicate GPU detected: rank 0 and rank 1 both on CUDA device`

**Root Cause**:
- Simulator mode uses `--device cuda` without local_rank assignment
- All ranks attempt to use the same GPU

---

## 4. Recommendations

### Short-term Fixes

1. **Fix channel mismatch in SVD wrapper**:
   - The `StableVideoUNet` should implement proper scheduler step logic
   - Store the 4-channel output and combine with next timestep's noise internally

2. **Add local_rank device assignment**:
   ```python
   local_rank = int(os.environ.get("LOCAL_RANK", 0))
   device = torch.device(f"cuda:{local_rank}")
   ```

3. **Reduce memory footprint**:
   - Use smaller latent resolution for testing: `1 8 14 32 32`
   - Enable gradient checkpointing if available
   - Consider model offloading between steps

### Long-term Improvements

1. **Integrate diffusers scheduler**:
   - Use EulerDiscreteScheduler for proper denoising step computation
   - Handle the noise prediction -> latent update correctly

2. **Multi-sample pipeline filling**:
   - Implement overlapping execution for multiple samples
   - Each GPU should process different samples in parallel

3. **Performance profiling**:
   - Add CUDA events for accurate timing
   - Profile communication vs computation ratio

---

## 5. Pipeline Logic Verification Summary

| Test | Status | Notes |
|------|--------|-------|
| Single process execution | PASS | All 28 steps complete |
| Multi-process communication | PASS | Latent tensors transferred correctly |
| Step assignment | PASS | Steps distributed evenly across ranks |
| Final output generation | PASS | Rank N-1 produces final latent |
| Tensor shape preservation | PASS | Shapes match throughout pipeline |

---

## 6. Appendix: Sample Logs

### 7-Process Pipeline Execution
```
[rank=0] step 27 completed in 22.49 ms
[rank=0] step 26 completed in 21.36 ms
[rank=0] step 25 completed in 14.14 ms
[rank=0] step 24 completed in 12.21 ms
[rank=0] sending latent to rank 1
[rank=1] received latent
[rank=1] step 23 completed in 22.82 ms
...
[rank=6] step 0 completed in 22.40 ms
[rank=6] final rank completed
Final latent norm: 11545.796875
```

---

## Conclusion

The pipeline parallel infrastructure is correctly implemented and verified using the DummyUNet in simulator mode. The production mode with the real SVD model requires additional work to:

1. Fix the scheduler integration for proper denoising
2. Implement per-rank GPU device assignment
3. Optimize memory usage for full resolution inference

The core pipeline logic (step splitting, latent communication, rank coordination) is functional and ready for integration with a properly configured video diffusion model.
