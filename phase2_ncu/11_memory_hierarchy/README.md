# Exercise 11: Memory Hierarchy Analysis

## Learning Goal

Use ncu to analyze L1 cache, L2 cache, and HBM (DRAM) hit rates for different memory access patterns. Understand the memory hierarchy on NVIDIA GPUs.

## Prerequisite

Exercise 06 (coalesced vs strided), Exercise 07 (compute/memory bound)

## Key Concepts

- GPU memory hierarchy: L1 (per-SM, ~128 KB) → L2 (shared, ~50 MB on H100) → HBM (DRAM, ~80 GB)
- L1 is fastest (~200 cycles), L2 is medium (~500 cycles), HBM is slowest (~800 cycles)
- Repeated access to same data increases cache hit rate
- Streaming access (no reuse) hits HBM directly
- ncu provides cache hit/miss rates per memory level

## Kernels

| Kernel | Pattern | Expected Cache Behavior |
|--------|---------|------------------------|
| `l1_hit` | Each thread reads one element, reuses it 128× in registers | High L1 hit rate (~85-95%), low HBM traffic |
| `l2_hit` | Each block processes a 4096-element strip in 16 passes via shared memory | Moderate L2 hit rate (~60-80%), some HBM |
| `hbm_only` | Streaming: each element read once, no reuse | Very low L1/L2 hit rate, dominated by HBM traffic |

## Build & Run

```bash
make -C ../.. phase2_ncu/11_memory_hierarchy/memory_hierarchy
./memory_hierarchy
```

## Profile with ncu

```bash
bash profile.sh
```

## ncu Flags

`--set full` captures all memory hierarchy metrics.

## How to Read ncu Results

### ncu GUI

1. Open with: `ncu-ui ./report11.ncu-rep`
2. For each kernel, check the **Memory Workload Analysis** section
3. Key metrics:
   - `l1tex__t_sectors_lookup_lookup_hit` / `l1tex__t_sectors_lookup_lookup_miss` → L1 hit/miss
   - `lts__t_sectors_lookup_lookup_hit` / `lts__t_sectors_lookup_lookup_miss` → L2 hit/miss
   - `dram__sectors_read` / `dram__sectors_write` → HBM traffic
4. Also check the **Memory Chart** for a visual breakdown

### Expected Findings

- **l1_hit**: high L1 hit rate (~85-95%), low HBM traffic — each element loaded once, reused 128× in registers
- **l2_hit**: moderate L2 hit rate (~60-80%), some HBM — 4096-element strip fits in L2, reused across 16 passes
- **hbm_only**: very low L1/L2 hit rate, dominated by HBM traffic — 16 MB array streamed through with no reuse

## Experiment

Vary the working set size to find the L1 and L2 cache capacity cliffs:

1. Change `N` to `1 << 18` (1 MB) — all kernels should fit in L2, observe hit rate changes
2. Change `N` to `1 << 24` (64 MB) — exceeds L2, observe HBM traffic increase
3. In `l2_hit`, vary the strip size (4096 → 8192 → 16384) to find the L2 capacity cliff
4. In `l1_hit`, vary the loop count (128 → 256 → 512) to see how register reuse affects L1 pressure