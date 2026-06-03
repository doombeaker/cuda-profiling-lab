# Exercise 06: Kernel Launch Overhead

## Learning Goal

Measure kernel launch overhead and see it on the nsys timeline. Understand that launching many small kernels can waste GPU time.

## Prerequisite

Exercise 01-04

## Key Concepts

- Kernel launch is NOT free — the CPU must submit work to the GPU via the CUDA driver
- Launch overhead is roughly constant per launch (typically 5-15 microseconds on modern GPUs)
- For small kernels, launch overhead can exceed actual computation time
- cudaDeviceSynchronize waits for all work to finish — visible as a gap on the timeline

## Build & Run

make && ./profile.sh

## nsys Flags Explained

Default --trace=cuda (no extra flags needed) — shows kernel launches and CUDA API calls

## How to Read Results

### Command-line Output

The program prints timing for each launch configuration. Compare 1 launch vs 100 launches — the overhead is visible.

### nsys GUI

1. Open nsys-ui ./report06.nsys-rep
2. Zoom into the CUDA API row — you'll see many blue cudaLaunchKernel API calls
3. On the GPU timeline, see the thin green kernel bars. For noop kernels, they're almost invisible — the API call time dominates
4. Compare: API call duration vs kernel duration. For noop kernels, API >> kernel time.

## Experiment

Try adding a kernel that spins in a loop for N iterations — find the grid size where compute time exceeds launch overhead.