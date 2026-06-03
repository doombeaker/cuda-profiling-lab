# Exercise 12: Warp Stall Reasons

## Learning Goal

Use ncu's **Warp State Stats** to identify WHY warps are stalled. Understand the 3 most common stall reasons: memory latency (Long Scoreboard), math latency (Short Scoreboard), and barrier synchronization.

## Prerequisite

Exercise 05 (occupancy), Exercise 07 (compute/memory bound)

## Key Concepts

- **Long Scoreboard (LG)**: warp waiting for global memory load. Most common stall in memory-bound kernels.
- **Short Scoreboard (SB)**: warp waiting for math/texture/shared memory result. Common in compute-bound kernels.
- **Barrier**: warp waiting at `__syncthreads()` for other warps in the block. Common in shared memory codes with heavy synchronization.
- **Sleep**: warp is inactive (no work issued for it).
- **Selected**: warp is actually executing instructions.

## The Three Kernels

| Kernel | Stall Reason | Description |
|--------|-------------|-------------|
| `memory_stall` | Long Scoreboard (LG) | Pure memory-bound: `c[i] = a[i] + b[i]`. Every instruction depends on a global memory load. |
| `math_stall` | Short Scoreboard (SB) | Compute-heavy: 500 iterations of `sinf`/`cosf` per element. Dependent math chain creates SB stalls. |
| `sync_stall` | Barrier | Heavy `__syncthreads()`: 24 barriers per block. Warps wait for peers to reach the barrier. |

All kernels operate on 4M elements with 256 threads per block.

## Build & Run

```bash
make && ./profile.sh
```

## ncu Flags

`--set full` captures Warp State Stats with all stall reasons.

## How to Read ncu Results

### ncu GUI

1. Open `ncu-ui ./report12.ncu-rep`
2. For each kernel, go to **Warp State Statistics** section
3. The pie chart shows cycle breakdown:
   - `memory_stall`: dominated by "Long Scoreboard" (60-80%)
   - `math_stall`: dominated by "Short Scoreboard" (40-60%)
   - `sync_stall`: significant "Barrier" percentage (20-40%)
4. Check the detailed table with all warp states as percentages

### Expected Findings

- `memory_stall`: Stall Long Scoreboard >> others
- `math_stall`: Stall Short Scoreboard dominates, some Stall LG
- `sync_stall`: Stall Barrier high, mixed with other stalls

## Experiment

Vary the math loop iterations and `__syncthreads` count to see how stall percentages change.