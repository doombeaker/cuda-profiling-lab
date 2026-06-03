#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/reduction
make -C ../.. $EXE
$NSYS profile -o ./report12 --force-overwrite true --trace=cuda $EXE
echo "Done. Open with: nsys-ui ./report12.nsys-rep. Compare the two reduction kernels on the GPU timeline."