#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/matmul
make -C ../.. $EXE
$NSYS profile -o ./report02 --force-overwrite true $EXE
echo "Done. Open with: nsys-ui ./report02.nsys-rep"
