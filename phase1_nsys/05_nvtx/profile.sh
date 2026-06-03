#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/nvtx
make -C ../.. $EXE
$NSYS profile -o ./report05 --force-overwrite true --trace=cuda,nvtx $EXE
echo "Done. Open with: nsys-ui ./report05.nsys-rep. Look for the NVTX row on the timeline."