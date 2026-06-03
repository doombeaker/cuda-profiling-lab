#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/cublas
make -C ../.. $EXE
$NSYS profile -o ./report08 --force-overwrite true --trace=cuda,cublas $EXE
echo "Done. Open with: nsys-ui ./report08.nsys-rep. Find the cuBLAS kernel on the GPU timeline."