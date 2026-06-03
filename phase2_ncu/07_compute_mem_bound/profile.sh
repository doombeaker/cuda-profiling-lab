#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/compute_mem_bound
make -C ../.. $EXE
$NCU --set full -o ./report07 --force-overwrite true $EXE
echo "Done. Open with: ncu-ui ./report07.ncu-rep"
