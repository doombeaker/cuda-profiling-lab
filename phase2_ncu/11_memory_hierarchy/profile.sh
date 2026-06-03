#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/memory_hierarchy
make -C ../.. $EXE
$NCU --set full -o ./report11 --force-overwrite true $EXE
echo "Open with: ncu-ui ./report11.ncu-rep. Compare L1/L2/HBM hit rates across kernels in Memory Workload Analysis."