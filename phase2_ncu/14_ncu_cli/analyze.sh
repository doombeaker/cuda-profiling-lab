#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/ncu_cli

echo "=== ncu --print-summary (summary table) ==="
$NCU --print-summary $EXE
echo ""

echo "=== ncu --print-summary per-kernel ==="
$NCU --print-summary per-kernel $EXE
echo ""

echo "=== ncu --csv (CSV output, showing first 5 lines) ==="
$NCU --csv $EXE 2>&1 | head -5
echo ""

echo "=== ncu --list-sets (available metric sets) ==="
$NCU --list-sets | head -15
echo ""

echo "=== ncu --list-sections (available sections) ==="
$NCU --list-sections | head -15
echo ""

echo "Done. Try: ncu --csv $EXE | grep saxpy"