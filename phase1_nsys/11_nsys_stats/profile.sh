#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/nsys_stats
make -C ../.. $EXE
$NSYS profile -o ./report11 --force-overwrite true --trace=cuda --stats=true $EXE
echo "Report generated. Run: source analyze.sh to see nsys stats reports."