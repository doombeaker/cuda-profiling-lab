# Exercise 10: Unified Memory Profiling

## Learning Goal

Use nsys to observe Unified Memory page faults and data migration. Understand when UVM helps and when it hurts performance.

## Prerequisite

Exercise 04 (bandwidth), Exercise 07 (pinned vs pageable)

## Key Concepts

- **Unified Memory (cudaMallocManaged)**: single pointer accessible from both CPU and GPU. The CUDA driver migrates pages on demand.
- **Page fault**: when GPU accesses a page resident on CPU (or vice versa), the driver migrates it. Migration latency ≈ PCIe transfer time.
- On the nsys timeline, page faults appear as "Unified Memory" events in a dedicated row.
- **Prefetch (cudaMemPrefetchAsync)**: hint to the driver to migrate pages BEFORE access, avoiding page fault latency during kernels.
- UVM is convenient but performance varies: prefer explicit memcpy for predictable performance; UVM for rapid prototyping.

## Build & Run

```
make && ./profile.sh
```

## nsys Flags Explained

`--trace=cuda,unified-memory` adds the Unified Memory row to the timeline

## How to Read Results

### Command-line Output

CPU-init→GPU (cold access) should be slower than GPU-init→GPU (hot access) due to page migration cost. Prefetch should be close to GPU-init timing.

### nsys GUI

1. Open `nsys-ui ./report10.nsys-rep`
2. Find the **Unified Memory** row on the timeline — this is only visible with `--trace=cuda,unified-memory`
3. For the CPU-init→GPU kernel: look for "page fault" and "data migration" events during kernel execution
4. For the GPU-init→GPU kernel: no migration events — pages are already resident
5. For the Prefetch case: migration happens BEFORE the kernel (during `cudaMemPrefetchAsync` call), so kernel runs with no faults

## Experiment

Try different data sizes and observe how migration overhead scales. Add a case with concurrent CPU/GPU access.