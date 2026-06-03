# Exercise 05: NVTX Timeline Annotation

## Learning Goal

Use NVTX (NVIDIA Tools Extension) to annotate CPU-side phases so they appear as labeled ranges on the nsys timeline. This makes it easy to map application logic to the profiler view.

## Prerequisite

Exercise 01-04: basic nsys profiling and timeline reading

## Key Concepts

- **nvtxDomainMarkA**: Places a colored, labeled mark (instantaneous event) on the timeline. Useful for flagging phase boundaries or milestones. Each mark carries an ARGB color and an ASCII message.
- **nvtxRangePushA / nvtxRangePop**: Creates a labeled time range on the timeline. Push starts the range, Pop ends it. Ranges nest properly (LIFO stack). In nsys-ui these appear as colored blocks on the NVTX row.
- **Color coding**: Assigning distinct colors to phases (green=Initialize, red=Compute, blue=Verify) makes it trivial to visually separate application stages at a glance.
- **Why annotating matters**: For complex applications with many phases, raw CUDA API traces are hard to interpret. NVTX annotations let you overlay your application's logical structure directly onto the GPU timeline, making it easy to correlate application phases with GPU activity and identify which phase is the bottleneck.

## Build & Run

```bash
make && ./profile.sh
```

## nsys Flags Explained

`--trace=cuda,nvtx` enables both CUDA API tracing and NVTX annotation capture. Without `nvtx` in the trace list, your NVTX ranges and marks will not appear in the report.

## How to Read Results

### Command-line Output

The program prints the GPU compute time (H2D + kernel + D2H) measured by CUDA events, and confirms that the vector_scale kernel correctly multiplied all elements by 2.0.

### nsys GUI

1. Open `nsys-ui ./report05.nsys-rep`
2. Expand the "NVTX" row on the timeline — you should see three colored ranges labeled **Initialize**, **Compute**, **Verify**
3. Inside the Compute range, locate the H2D copy, kernel, and D2H copy on the CUDA rows
4. This shows how NVTX lets you correlate application phases with GPU activity

## Experiment

Try adding more granular NVTX ranges inside the "Compute" phase to separate H2D, kernel, and D2H. This gives you per-subphase timing directly on the timeline.