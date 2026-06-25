#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/bank_conflicts
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REL=${SCRIPT_DIR#"$PROJECT_ROOT/"}
make -C "$PROJECT_ROOT" "$REL/$(basename "$EXE")"
$NCU --set full -o ./report09 --force-overwrite true $EXE
echo "Done. Open with: ncu-ui ./report09.ncu-rep"
echo "Check Memory Workload Analysis for bank conflict metrics."