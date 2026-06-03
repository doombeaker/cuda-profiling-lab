# Exercise 09: Shared Memory Bank Conflicts

## Learning Goal

Use ncu to detect shared memory bank conflicts. Understand how bank conflicts degrade shared memory bandwidth.

## Prerequisite

Exercise 06 (coalesced vs strided memory)

## Key Concepts

- Shared memory has 32 banks (one per 4-byte word). Threads in a warp access banks concurrently.
- **No conflict**: each thread accesses a different bank → single-cycle access.
- **2-way conflict**: 2 threads share a bank → 2 cycles (serialized).
- **32-way conflict**: all 32 threads share a bank → 32 cycles (worst case).

### Bank Mapping

For a shared memory array `smem[256]` of floats (4 bytes each):
- `smem[0]` → bank 0, `smem[1]` → bank 1, ..., `smem[31]` → bank 31
- `smem[32]` → bank 0, `smem[33]` → bank 1, ... (wraps around every 32 elements)

### Kernels in This Exercise

| Kernel | Read Pattern | Bank Conflict | Expected shared_efficiency |
|--------|-------------|---------------|---------------------------|
| `no_conflict` | `smem[tid]` (stride-1) | None | ~100% |
| `conflict_2way` | `smem[(tid*2) % 256]` (stride-2) | 2-way | ~50% |
| `conflict_32way` | `smem[0]` (all same bank) | 32-way | ~3.1% |

## Build & Run

```bash
make && ./profile.sh
```

Or build and run standalone:

```bash
make -C ../.. phase2_ncu/09_bank_conflicts/bank_conflicts
./bank_conflicts
```

## ncu Flags

`--set full` includes shared memory bank conflict metrics.

## How to Read ncu Results

### ncu GUI

1. Open `ncu-ui ./report09.ncu-rep`
2. For each kernel, go to **Memory Workload Analysis** section
3. Find: `shared_load_bank_conflicts`, `shared_store_bank_conflicts`
4. Also check: `shared_efficiency` (should be 100% for no-conflict, ~50% for 2-way, ~3% for 32-way)

### Expected Findings

- **no_conflict**: `shared_efficiency` ≈ 100%, bank conflicts = 0
- **conflict_2way**: `shared_efficiency` ≈ 50%, 2 replays per access
- **conflict_32way**: `shared_efficiency` ≈ 3.1%, 32 replays per access

## Experiment

Try stride-4 and stride-8 to see 4-way and 8-way conflicts. Modify `conflict_2way`:

```cuda
int idx4 = (tid * 4) % 256;   // 4-way conflict
int idx8 = (tid * 8) % 256;   // 8-way conflict
```

## Why This Matters

Shared memory is a programmer-managed cache with ~100x lower latency than global memory. But bank conflicts serialize accesses within a warp, effectively dividing bandwidth by the conflict degree. A 32-way conflict turns shared memory into a bottleneck worse than L2 cache. Always check `shared_efficiency` in ncu when using shared memory heavily.