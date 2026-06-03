#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/kernel_launch_overhead
make -C ../.. $EXE
$NSYS profile -o ./report06 --force-overwrite true $EXE
echo "Done. Open with: nsys-ui ./report06.nsys-rep. Zoom into the CUDA API row to see kernel launch calls."