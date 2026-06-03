# Exercise 07: Pinned vs Pageable Memory

## Learning Goal

Understand why `cudaMallocHost` (pinned memory) is critical for PCIe transfer performance. See the staging copy on the nsys timeline.

## Prerequisite

Exercise 01 (basic timeline), Exercise 04 (bandwidth measurement)

## Key Concepts

- **Pageable memory (malloc)**: OS may swap pages, so CUDA must first copy to a pinned staging buffer before DMA transfer. This staging copy is visible on the nsys timeline as an extra H2D operation.
- **Pinned memory (cudaMallocHost)**: memory is locked in physical RAM, allowing GPU DMA engine to read directly at full PCIe bandwidth (~32 GB/s for PCIe Gen4 x16).
- The nsys timeline clearly shows the difference: pageable transfers show a 2-step process, pinned transfers show a single fast transfer.
- **Cost**: pinned memory reduces available system RAM for the OS, so use sparingly.

## Build & Run

```
make && ./profile.sh
```

## nsys Flags Explained

Default `--trace=cuda` shows all CUDA API and memory operations.

## How to Read Results

### Command-line Output

Compare bandwidth numbers — pinned should be 2-10x faster than pageable.

### nsys GUI

1. Open `nsys-ui ./report07.nsys-rep`
2. Look at the CUDA memory operations row
3. For pinned: single blue H2D bar, fills the transfer time
4. For pageable: may show TWO memory operations — a staging copy (pageable→pinned buffer) then the actual DMA transfer
5. The total pageable H2D time is longer due to the staging step

## Experiment

Try different transfer sizes (`1<<24`, `1<<26`, `1<<28`) and see how the bandwidth gap changes.