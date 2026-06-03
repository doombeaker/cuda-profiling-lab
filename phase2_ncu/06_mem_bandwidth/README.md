# Exercise 06: Memory Bandwidth — Coalesced vs Strided Access

## Goal

Understand how memory access patterns affect bandwidth and learn to read ncu memory metrics.

## Background

GPUs achieve peak bandwidth when threads in a warp access **contiguous** memory locations (coalesced access). When adjacent threads access scattered addresses, the GPU issues separate memory transactions for each cache line, wasting bandwidth. This exercise compares two kernels that perform the same computation but with different access patterns.

## Kernels

| Kernel | Pattern | What happens |
|--------|---------|-------------|
| `kernel_coalesced` | `data[i]` | Adjacent threads read/write adjacent 4-byte floats → single 128-byte transaction per warp |
| `kernel_strided` | `data[(i*32) % n]` | Adjacent threads hit addresses 128 bytes apart → 32 separate transactions per warp |

## Key ncu Metrics

After profiling, compare these metrics between the two kernels:

| Metric | Section in ncu-ui | What to look for |
|--------|-------------------|-------------------|
| **Global Load Efficiency** | Memory > Global Load | Coalesced ≈ 100%, Strided ≈ 3–12% |
| **Global Store Efficiency** | Memory > Global Store | Coalesced ≈ 100%, Strided ≈ 3–12% |
| **L1/TEX Cache Hit Rate** | Memory > L1/TEX Cache | Strided may show higher hit rate due to re-accessed lines |
| **L2 Cache Hit Rate** | Memory > L2 Cache | Compare cache behavior between patterns |
| **DRAM Throughput** | Memory > DRAM | Strided kernel forces more DRAM transactions |
| **L2 Throughput** | Memory > L2 | Shows how much data passes through L2 |

## Instructions

1. Build and run:
   ```bash
   make -C ../.. phase2_ncu/06_mem_bandwidth/mem_bandwidth
   ./mem_bandwidth
   ```

2. Profile with ncu:
   ```bash
   bash profile.sh
   ```

3. Open the report:
   ```bash
   ncu-ui ./report06.ncu-rep
   ```

4. Compare **Global Load Efficiency** and **Global Store Efficiency** between `kernel_coalesced` and `kernel_strided`. The coalesced kernel should show efficiencies near 100%, while the strided kernel will be dramatically lower because each warp issues 32 separate 128-byte transactions instead of one.

## Why This Matters

Most GPU kernels are memory-bound. Coalescing is the single most impactful optimization for such kernels — it can improve bandwidth utilization by 10–30x without changing the algorithm at all. Always check global load/store efficiency in ncu when profiling memory-bound code.
