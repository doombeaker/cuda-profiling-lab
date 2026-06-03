# Exercise 12: Parallel Reduction

## Learning Goal

Implement parallel reduction with shared memory. Compare interleaved vs sequential addressing patterns and observe block synchronization on the nsys timeline. This is a capstone exercise combining timeline analysis, kernel duration comparison, and memory access pattern understanding.

## Prerequisite

All previous Phase 1 exercises (01-11)

## Key Concepts

- **Parallel reduction**: sum N elements using a tree-based approach in parallel. O(log N) steps.
- **Shared memory**: `__shared__` memory is on-chip, ~100x faster than global memory, used for intra-block communication.
- **`__syncthreads()`**: barrier — all threads in a block must reach this point before any can proceed. Visible on nsys timeline as synchronization points.
- **Interleaved addressing (bad)**: threads with large stride lead to warp divergence — half the warps are idle in later stages.
- **Sequential addressing (good)**: threads within a warp access consecutive elements, no divergence, full warp utilization.
- **Two-pass reduction**: first pass reduces to grid_size partial sums, then a small pass (or CPU) combines them.

## Build & Run

```bash
make && ./profile.sh
```

## nsys Flags Explained

`--trace=cuda` captures all kernel launches and their durations

## How to Read Results

### Command-line Output

Sequential addressing should be measurably faster than interleaved (10-30% typical).

### nsys GUI

1. Open `nsys-ui ./report12.nsys-rep`
2. On the GPU timeline, find `reduce_interleaved` and `reduce_sequential` kernels
3. Compare their durations — sequential should be shorter
4. Hover over each kernel to see block count (grid_size = 65536 blocks)
5. Note: for such a large grid, kernel launch overhead is negligible compared to compute time
6. Observe the H2D copy (blue bar) and D2H copy for partial results

## Experiment

1. Vary block size (128, 256, 512) and observe impact on reduction time
2. Add a third version using warp-level primitives (`__shfl_down_sync`) for the last few reduction steps
3. Compare with cuBLAS reduction (`cublasSasum`) if curious