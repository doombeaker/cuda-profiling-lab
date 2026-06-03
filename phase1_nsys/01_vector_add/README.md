# Exercise 01: Vector Add

## What This Exercise Teaches

This exercise introduces **Nsight Systems (nsys)** profiling for a basic CUDA vector addition. You will learn:

- **Basic nsys profile usage**: How to launch a CUDA program under `nsys profile` to capture a timeline
- **Reading the timeline**: The three segments of a CUDA program are clearly visible:
  1. **H2D copy** — Host-to-Device memory transfers (`cudaMemcpy` with `cudaMemcpyHostToDevice`)
  2. **Kernel execution** — The `vector_add` kernel running on the GPU
  3. **D2H copy** — Device-to-Host memory transfer (`cudaMemcpy` with `cudaMemcpyDeviceToHost`)
- **Identifying bottlenecks**: In a simple program like this, memory transfers often dominate. The timeline makes this immediately obvious.

## Instructions

### Build

```bash
make
```

### Profile

```bash
./profile.sh
```

This runs the program under `nsys profile` and produces `report01.nsys-rep`.

### View Results

Open the report with the Nsight Systems GUI:

```bash
nsys-ui ./report01.nsys-rep
```

Look for the three segments (H2D → Kernel → D2H) on the GPU timeline.
