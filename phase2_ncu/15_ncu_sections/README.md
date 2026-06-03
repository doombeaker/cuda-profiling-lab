# Exercise 15: ncu Sections & Targeted Profiling

## Learning Goal

Learn to select specific ncu sections/metrics instead of always using `--set full`. Use `--section` for targeted profiling and understand when to use `--set analysis` vs `--set full`.

## Prerequisite

All previous ncu exercises

## Key Concepts

- `--set full` collects ALL metrics — comprehensive but slow (2-5+ seconds per kernel launch)
- `--set analysis` collects the most commonly useful metrics — faster (~1 second per launch)
- `--section <id>` collects only one specific section — fastest, most targeted
- Common section IDs: SpeedOfLight, Occupancy, MemoryWorkloadAnalysis, WarpStateStats, SourceCounters, SchedulerStats
- Use `ncu --list-sections` to see all available sections for your GPU
- For production profiling: start with `--set analysis`, then drill down with `--section` for specific issues

## Build & Run

```bash
make
./profile.sh        # profiles with --set analysis (fast)
```

### Try These Variations (manual)

```bash
NCU=/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu
EXE=./ncu_sections

# 1. Speed of Light only (fastest, high-level overview)
$NCU --section SpeedOfLight --print-summary $EXE

# 2. Memory Workload (targeted at memory metrics)
$NCU --section MemoryWorkloadAnalysis --print-summary $EXE

# 3. Occupancy only
$NCU --section Occupancy --print-summary $EXE

# 4. Compare --set analysis vs --set full timing
time $NCU --set analysis $EXE
time $NCU --set full $EXE
```

## ncu Flags Explained

- `--set full`: all metrics (slow, comprehensive) — best for initial exploration
- `--set analysis`: essential metrics (faster) — best for routine profiling
- `--section SpeedOfLight`: just the Speed of Light summary — best for bottleneck identification
- `--section Occupancy`: just occupancy data — best for launch configuration tuning
- `--section MemoryWorkloadAnalysis`: just memory metrics — best for memory-bound kernels

## How to Use in Practice

1. First run: `ncu --set analysis` — get the big picture
2. If memory-bound → `ncu --section MemoryWorkloadAnalysis` — drill into memory details
3. If occupancy is low → `ncu --section Occupancy` — check register/barrier limits
4. If warps are stalling → `ncu --section WarpStateStats` — find stall reasons
5. Only use `--set full` for comprehensive debugging or when you need every metric

## Experiment

Run `ncu --list-sections` and try profiling with a section you haven't used yet (e.g., InstructionStatistics).