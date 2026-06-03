#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/warp_divergence
make -C ../.. $EXE
$NCU --set full -o ./report08 --force-overwrite true $EXE
echo "Open with: ncu-ui ./report08.ncu-rep. Compare branch efficiency across kernels."