#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/launch_config
make -C ../.. $EXE
$NCU --set full -o ./report10 --force-overwrite true $EXE
echo "Done. Open with: ncu-ui ./report10.ncu-rep. Compare occupancy across the 5 launches."