# Exercise 05: Occupancy Analysis with NCU

## Objective

Learn how to use NVIDIA Nsight Compute (`ncu`) with `--set full` to analyze kernel occupancy, and understand how register pressure reduces occupancy.

## Background

**Occupancy** is the ratio of active warps per SM to the maximum number of warps that can reside on an SM simultaneously. Higher occupancy means more warps are available to hide latency, but it is not the only factor in performance.

Two key occupancy metrics:
- **Theoretical Occupancy**: Calculated from kernel resource usage (registers per thread, shared memory per block). This is what the compiler and driver compute before launch.
- **Achieved Occupancy**: The actual occupancy measured during execution, which may be lower than theoretical due to scheduling and other runtime factors.

The primary limiter of occupancy is **register usage per thread**. Each SM has a fixed register file (e.g., 65536 registers on modern GPUs). More registers per thread means fewer threads can reside on an SM, reducing occupancy.

## The Three Kernels

| Kernel | Local Variables | Register Pressure | Expected Occupancy |
|--------|----------------|-------------------|--------------------|
| `kernel_low_reg` | Minimal (just index + result) | Low | Highest |
| `kernel_medium_reg` | 40 local floats | Medium | Medium |
| `kernel_high_reg` | 80 local floats | High | Lowest |

All three kernels operate on the same data (16M elements) with the same grid/block configuration (`<<<65536, 256>>>`). The `medium_reg` and `high_reg` kernels use `__launch_bounds__(256)` to hint the compiler about the block size.

## Profiling with NCU

Run the profiling script:

```bash
./profile.sh
```

This invokes `ncu --set full`, which collects all available metrics including occupancy data.

## What to Look For

### 1. GPU Speed of Light Section

This section gives a high-level overview showing compute and memory utilization. Compare the SM utilization across the three kernels — the low register kernel should show higher utilization.

### 2. Occupancy Section

This is the key section for this exercise. For each kernel, examine:

- **Theoretical Occupancy**: The percentage of maximum warps that can be active on an SM given the kernel's resource requirements.
- **Achieved Occupancy**: The actual measured occupancy during execution.
- **Registers Per Thread**: The number of registers the compiler allocated per thread.
- **Occupancy Limiting Factor**: What resource is preventing higher occupancy (typically registers).

### 3. The Inverse Relationship

You should observe a clear inverse relationship:

```
Registers/Thread ↑  →  Warps/SM ↓  →  Occupancy ↓
```

- `kernel_low_reg`: Few registers → many warps can reside on SM → high occupancy
- `kernel_medium_reg`: More registers → fewer warps fit → lower occupancy
- `kernel_high_reg`: Many registers → very few warps fit → lowest occupancy

## Key Takeaway

Register pressure directly limits occupancy. When a kernel uses many registers, fewer threads can run concurrently on each SM, which reduces the GPU's ability to hide memory and instruction latency. Use `ncu` to identify when register pressure is your occupancy bottleneck, and consider:
- Reducing local variable count
- Using `__launch_bounds__` to guide register allocation
- Splitting complex kernels into simpler ones
