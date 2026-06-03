#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/mem_bandwidth
make -C ../.. $EXE
$NCU --set full -o ./report06 --force-overwrite true $EXE
echo "Done. Open with: ncu-ui ./report06.ncu-rep"
