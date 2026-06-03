# Exercise 03: Multi-Stream Overlap

## What This Exercise Teaches

This exercise demonstrates **CUDA stream concurrency** and how to verify it on the Nsight Systems timeline. You will learn:

- **Async memory transfers**: Using `cudaMemcpyAsync` instead of synchronous `cudaMemcpy` so that transfers do not block the host
- **Stream-level concurrency**: Launching independent H2D → Kernel → D2H pipelines across 4 CUDA streams so they can overlap on the GPU
- **Reading the multi-stream timeline**: On the nsys timeline, each stream appears as its own row. You should see:
  1. **Stream 7** (default) — idle while the 4 streams execute
  2. **Stream 13–16** (or similar) — each showing its own H2D copy, kernel, and D2H copy
  3. **Overlap** — the kernel on one stream running while a memcpy on another stream is in flight
- **The value of concurrency**: Compared to a single-stream version, overlapping computation and transfers across streams can significantly reduce total wall-clock time

## Instructions

### Build

```bash
make
```

### Profile

```bash
./profile.sh
```

This runs the program under `nsys profile` with `--trace=cuda,nvtx` and produces `report03.nsys-rep`.

### View Results

Open the report with the Nsight Systems GUI:

```bash
nsys-ui ./report03.nsys-rep
```

### What to Look For

1. **Identify each stream's row** on the GPU timeline — you should see 4 separate stream rows
2. **Check for overlap** — the kernel on one stream should be executing concurrently with memcpy operations on other streams
3. **Compare with a single-stream approach** — if all operations were serialized, the total time would be roughly 4× the single-chunk time; with overlap, the total should be noticeably less
