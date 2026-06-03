# Exercise 10: Launch Configuration & Block Size Tuning

## Learning Goal

Use ncu to compare different block sizes (launch configurations) for the same kernel. Learn how block size affects occupancy and performance.

## Prerequisite

Exercise 05 (occupancy)

## Key Concepts

- **Block size affects occupancy**: too small → SMs underutilized (hardware limit on max blocks per SM). Too large → register pressure limits occupancy.
- **Dynamic shared memory**: `extern __shared__` + third kernel launch parameter
- **ncu can profile multiple launches in one run** — just launch all configs in the same program

## Build & Run

```bash
make && ./profile.sh
```

## ncu Flags

`--set full` captures occupancy, register, and shared memory data for each launch.

## How to Read Results

### Command-line

The program prints timing for each block size. Look for the fastest.

### ncu GUI

1. Open `ncu-ui ./report10.ncu-rep`
2. Each kernel launch appears separately (5 launches)
3. For each launch, check Occupancy section:
   - Theoretical Occupancy at different block sizes
   - Registers/Thread
4. The optimal block size balances occupancy with resource limits
5. Check the Occupancy limiting factors for each launch

### Expected Findings

- **Small blocks (32)**: high occupancy per-SM but limited total threads due to max blocks/SM
- **Medium blocks (128-256)**: sweet spot — good occupancy, good parallelism
- **Large blocks (512)**: limited by registers/shared memory per SM

## Experiment

Add a 1024 block size test and observe the occupancy drop due to register pressure.