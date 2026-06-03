# Exercise 14: ncu CLI Analysis

## Learning Goal

Learn to use the `ncu` command-line interface for headless profiling analysis. Use `ncu --print-summary` and `ncu --csv` to extract kernel metrics without opening the GUI.

## Prerequisite

All previous ncu exercises

## Key Concepts

- `ncu --print-summary`: prints a text summary table after each kernel run, showing key metrics (duration, registers, occupancy, etc.)
- `ncu --csv`: outputs results in CSV format for programmatic parsing (grep/awk/python)
- `ncu --print-gpu-trace`: shows kernel name and duration like a mini-timeline
- `ncu --list-sets`: lists available metric sets (full, analysis, basic, etc.)
- `ncu --list-sections`: lists available sections (SpeedOfLight, Occupancy, etc.)

## Build & Run

```bash
make
./profile.sh      # standard profile (generates report)
source analyze.sh # CLI analysis (4 different ncu modes)
```

## analyze.sh Contents

The analyze.sh script demonstrates 4 ncu CLI modes:

1. `ncu --print-summary`: summary table for each kernel
2. `ncu --print-summary per-kernel`: per-kernel breakdown
3. `ncu --csv`: CSV output that can be piped to grep
4. `ncu --list-sets` and `--list-sections`: discover available metrics

## How to Read Results

### ncu --print-summary output

Shows a table with columns: ID, Kernel Name, Duration (ns), Grid/Block, Registers, Occupancy, etc.
Compare the 3 kernels' durations and register usage directly in the terminal.

### ncu --csv output

Machine-readable format. Example: `ncu --csv ./ncu_cli | grep saxpy`

## Experiment

Try: `ncu --set analysis --print-summary ./ncu_cli` (uses faster, lighter "analysis" set)

Try: `ncu --section SpeedOfLight --print-summary ./ncu_cli` (targeted section only)