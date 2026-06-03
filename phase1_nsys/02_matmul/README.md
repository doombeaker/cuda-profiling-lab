# Exercise 02 — Matrix Multiply: Naive vs Tiled

## Goal

Compare two matrix multiplication kernels on the Nsight Systems timeline and observe the dramatic performance difference between a naive implementation and a shared-memory tiled implementation.

## Kernels

### Naive Kernel (`matmul_naive`)

Each thread computes one element of C by iterating over an entire row of A and an entire column of B. Every iteration fetches from global memory, resulting in **2N global memory accesses per output element**. For N=1024, that is 2048 global reads per C element.

### Tiled Kernel (`matmul_tiled`)

Each thread block cooperatively loads TILE×TILE sub-matrices of A and B into shared memory. Threads then compute partial dot products from shared memory, which is orders of magnitude faster than global memory. This reduces global memory traffic by a factor of TILE (16×), since each tile is loaded once and reused by TILE threads.

## Why Tiled Is Faster

- **Shared memory** is on-chip and has ~100× lower latency than global memory
- **Data reuse**: each tile element is read from global memory once but used TILE times
- **Memory coalescing**: contiguous threads read contiguous addresses when loading tiles
- The naive kernel performs N global reads per multiply-accumulate; the tiled kernel performs only 1/TILE global reads per multiply-accumulate

## Build & Run

```bash
make -C ../.. phase1_nsys/02_matmul/matmul
./matmul
```

## Profile with Nsight Systems

```bash
./profile.sh
nsys-ui ./report02.nsys-rep
```

## What to Look For on the Timeline

1. Two kernel launches: `matmul_naive` and `matmul_tiled`
2. The tiled kernel should execute significantly faster than the naive kernel
3. Hover over each kernel to compare duration — expect the tiled kernel to be several times faster
4. The memory throughput section will show higher effective bandwidth utilization for the tiled kernel
