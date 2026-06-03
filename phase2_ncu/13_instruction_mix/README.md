# Exercise 13: Instruction Mix Analysis

## Learning Goal

Use ncu to analyze the instruction type breakdown (FP32, FP64, INT, control flow, memory) of different kernels. Understand how pipeline utilization varies by instruction mix.

## Prerequisite

Exercise 12 (stall reasons)

## Key Concepts

- GPU SMs have separate pipelines: FP32 pipe, FP64 pipe (fewer units), INT pipe, load/store pipe, special function unit (SFU).
- A kernel dominated by FP32 uses mostly the FP32 pipe; one dominated by INT mostly uses the INT pipe.
- ncu shows instruction counts by type — see which pipelines are saturated and which are idle.
- Control flow instructions (branches, jumps) add overhead and reduce compute pipeline utilization.

## The Three Kernels

| Kernel | Instruction Mix | Description |
|--------|----------------|-------------|
| `fp32_only` | FP32 dominant | Pure FP32 arithmetic: `sinf`, `cosf`, `fmaf`, multiply, add. No integer math, no branches. |
| `int_heavy` | INT dominant | Heavy integer computation: LCG, XOR, bit rotation, modulo. Only FP32 at the edges (convert in/out). |
| `control_heavy` | Control/Branch dominant | 6-way branch per iteration × 200 iterations + modulo branch. FP32 arithmetic inside branches. |

All kernels operate on 4M elements with 256 threads per block.

## Build & Run

```
make && ./profile.sh
```

## ncu Flags

`--set full` captures instruction mix counters.

## How to Read ncu Results

### ncu GUI

1. Open `ncu-ui ./report13.ncu-rep`
2. Go to **Source Counters** or **Instruction Statistics** section
3. Look at:
   - Instructions by type: FP32, FP64, Integer, Control, Load/Store, Misc
   - For `fp32_only`: FP32 should dominate (70-90%)
   - For `int_heavy`: Integer should be high (40-60%)
   - For `control_heavy`: Control instructions should be notably higher than the others
4. Check **GPU Speed of Light** — does any pipeline show high utilization?

### Expected Findings

- `fp32_only`: FP32 instructions >> INT/Control
- `int_heavy`: INT >> FP32, significant bit manipulation ops
- `control_heavy`: Control/Branch >> others, FP32 also present but many branches

## Experiment

Add a kernel using `double` (FP64) and compare instruction counts. The FP64 pipeline has fewer units on most GPUs (e.g., 1/32 of FP32 throughput on consumer GPUs, 1/2 on H100).