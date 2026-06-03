# Exercise 08: Warp Divergence & Branch Efficiency

## Learning Goal

Use ncu to detect warp divergence. Understand branch efficiency and why divergent warps hurt SIMT performance.

## Prerequisite

Exercise 05 (occupancy), Exercise 12 reduction from nsys phase

## Key Concepts

- **SIMT execution**: all threads in a warp execute the same instruction. When threads diverge (if/else), the GPU serializes both paths.
- **Branch efficiency** = (active threads) / (total threads per warp) averaged over all executed instructions. 100% means no divergence.
- **Interleaved addressing** (`tid % 2s == 0`) causes divergence because warp members take different paths as stride grows.
- **Sequential addressing** (`tid < s`) preserves warp uniformity — all active threads in a warp execute the same path.

## The Three Kernels

| Kernel | Pattern | Expected Branch Efficiency | Why |
|--------|---------|---------------------------|-----|
| `divergent_reduce` | Interleaved: `tid % (2*s) == 0` | ~50-70% | Warp members diverge as stride grows; only 1/N threads active in later stages |
| `uniform_reduce` | Sequential: `tid < s` | ~100% | Consecutive threads take same branch; entire warps stay active or inactive together |
| `branch_threshold` | if/else on data value | ~50% | Half the elements are above threshold, half below; warps split execution |

## Build & Run

```bash
make && ./profile.sh
```

## ncu Flags

`--set full` captures all metrics including `branch_efficiency`, `warp_execution_efficiency`, and `avg_divergent_branches`.

## How to Read ncu Results

### ncu GUI

1. Open `ncu-ui ./report08.ncu-rep`
2. Compare the 3 kernels:
   - `divergent_reduce`: select kernel → Source Counters → look for **Branch Efficiency** (~50%, low)
   - `uniform_reduce`: Branch Efficiency should be ~100%
   - `branch_threshold`: look at **avg_divergent_branches** (should show divergence)
3. Check **Warp State Stats** → **Avg. Divergent Branches**

### Expected Findings

- `divergent_reduce`: ~50-70% branch efficiency, many divergent branches
- `uniform_reduce`: ~100% branch efficiency, zero divergent branches
- `branch_threshold`: ~50% branch efficiency (half the warps take each path)

## Experiment

Modify the reduction stride pattern to test different divergence severities. Try:
- Changing the interleaved stride from `s *= 2` to `s *= 4` (fewer active threads per step)
- Changing the threshold in `branch_threshold` to skew the branch ratio (e.g., 0.1 or 0.9)
- Observe how branch efficiency changes in ncu