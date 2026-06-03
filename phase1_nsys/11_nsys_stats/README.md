# Exercise 11: nsys stats — CLI Analysis

## Learning Goal

Learn to use `nsys stats` to extract profiling statistics from an `.nsys-rep` file without opening the GUI. This is essential for CI/CD pipelines and automated performance regression testing.

## Prerequisite

Exercise 01-04 (basic profiling with nsys)

## Key Concepts

- nsys generates `.nsys-rep` and `.sqlite` files. `nsys stats` reads these and produces structured reports.
- Common nsys stats reports:
  * `--report cuda_gpu_kern_sum` — kernel execution summary (name, time, invocations, etc.)
  * `--report cuda_gpu_mem_size_sum` — memory transfer summary
  * `--report cuda_api_sum` — CUDA API call summary
  * `--report cuda_gpu_trace` — full GPU trace in text format
- You can pipe nsys stats output to grep/awk for programmatic parsing

## Build & Run

```bash
make
./profile.sh          # generates report11.nsys-rep
source analyze.sh     # runs nsys stats reports
```

## nsys stats Reports Used

The `analyze.sh` script demonstrates 4 useful report types:

1. Kernel execution summary (names, counts, avg/min/max times)
2. Memory operation summary (H2D/D2H sizes and bandwidth)
3. CUDA API summary (API call counts and times)
4. Full GPU trace (not shown by default — uncomment to see)

## How to Read Results

### analyze.sh Output

Each nsys stats report prints a table. Look for:

- Kernel names and their average execution times
- Memory transfer sizes and throughput
- API call overhead

### Compare with GUI

Open `nsys-ui ./report11.nsys-rep` and verify that the nsys stats numbers match the timeline measurements.

## Experiment

Try: `nsys stats --report cuda_gpu_kern_sum --format csv,column | grep vector_add`

Try the `--format csv` option for machine-readable output.