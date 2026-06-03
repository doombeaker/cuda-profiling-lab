# Exercise 07: Compute-Bound vs Memory-Bound Kernels

## Goal

Learn to distinguish compute-bound from memory-bound kernels and read the **GPU Speed of Light** section in NCU to identify which resource limits performance.

## Key Concepts

### Arithmetic Intensity

Arithmetic intensity = FLOPs / Bytes transferred. It determines whether a kernel is limited by memory bandwidth or compute throughput.

| Kernel | Operation | FLOPs | Bytes | AI (FLOP/byte) | Bound by |
|---|---|---|---|---|---|
| `kernel_memory_bound` | `c[i] = a[i] + b[i]` | 1 | 12 (3×4) | 0.08 | Memory |
| `kernel_compute_bound` | 1024× transcendental ops | ~10,240 | 12 | ~800 | Compute |

### Roofline Model

The roofline model plots performance (FLOP/s) against arithmetic intensity. Two ceilings:

- **Memory ceiling**: `Peak FLOP/s = Bandwidth × AI` — a diagonal line rising with AI until it hits the compute roof.
- **Compute ceiling**: A horizontal line at peak FLOP/s of the GPU.

A kernel's position on this chart tells you its bottleneck:
- Low AI → sits under the memory slope → memory-bound
- High AI → sits under the compute roof → compute-bound

### GPU Speed of Light (NCU)

The **GPU Speed of Light** section in NCU shows three key metrics:

- **Memory %** — fraction of peak memory bandwidth utilized
- **Compute %** — fraction of peak compute throughput utilized
- **Occupancy %** — fraction of maximum warps resident on SMs

For `kernel_memory_bound`, expect **Memory % near 80-90%** and **Compute % very low**.
For `kernel_compute_bound`, expect **Compute % high** and **Memory % low**.

## What to Look For

1. Run `bash profile.sh` to generate the NCU report.
2. Open with `ncu-ui ./report07.ncu-rep`.
3. Select each kernel and check the **GPU Speed of Light** section.
4. The kernel hitting ~100% memory utilization is memory-bound.
5. The kernel hitting high compute utilization is compute-bound.
6. Switch to the **Roofline** chart tab — see where each kernel lands relative to the roofline.

## Running

```bash
bash profile.sh
ncu-ui ./report07.ncu-rep
```
