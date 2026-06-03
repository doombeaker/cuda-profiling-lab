#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/unified_memory
make -C ../.. $EXE
$NSYS profile -o ./report10 --force-overwrite true --trace=cuda,unified-memory $EXE
echo "Done. Open with: nsys-ui ./report10.nsys-rep"
echo "Look for the 'Unified Memory' row showing page faults and migrations."