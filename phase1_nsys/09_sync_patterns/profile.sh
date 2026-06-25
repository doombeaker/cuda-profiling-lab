#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
EXE=$(dirname $0)/sync_patterns
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REL=${SCRIPT_DIR#"$PROJECT_ROOT/"}
make -C "$PROJECT_ROOT" "$REL/$(basename "$EXE")"
$NSYS profile -o ./report09 --force-overwrite true --trace=cuda $EXE
echo "Done. Open with: nsys-ui ./report09.nsys-rep"
echo "Compare kernel overlap in the GPU section across the three sync patterns."