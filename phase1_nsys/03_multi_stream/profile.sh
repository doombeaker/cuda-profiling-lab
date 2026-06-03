#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/multi_stream
make -C ../.. $EXE
$NSYS profile -o ./report03 --force-overwrite true --trace=cuda,nvtx $EXE
echo "Done. Open with: nsys-ui ./report03.nsys-rep"
