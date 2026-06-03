#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/ncu_cli
make -C ../.. $EXE
$NCU --set full -o ./report14 --force-overwrite true $EXE
echo "Report generated. Run 'source analyze.sh' for ncu CLI analysis."