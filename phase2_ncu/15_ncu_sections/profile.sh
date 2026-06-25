#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NCU=$CUDA_HOME/nsight-compute-2024.1.1/ncu
EXE=$(dirname $0)/ncu_sections
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
REL=${SCRIPT_DIR#"$PROJECT_ROOT/"}
make -C "$PROJECT_ROOT" "$REL/$(basename "$EXE")"
$NCU --set analysis -o ./report15 --force-overwrite true $EXE
echo "Done. Open with: ncu-ui ./report15.ncu-rep (profiled with --set analysis)."
echo ""
echo "=== Try these manual --section experiments ==="
echo "NCU=$NCU"
echo "EXE=$EXE"
echo ""
echo "# 1. Speed of Light only (fastest, high-level overview)"
echo "\$NCU --section SpeedOfLight --print-summary \$EXE"
echo ""
echo "# 2. Memory Workload (targeted at memory metrics)"
echo "\$NCU --section MemoryWorkloadAnalysis --print-summary \$EXE"
echo ""
echo "# 3. Occupancy only"
echo "\$NCU --section Occupancy --print-summary \$EXE"
echo ""
echo "# 4. Compare --set analysis vs --set full timing"
echo "time \$NCU --set analysis \$EXE"
echo "time \$NCU --set full \$EXE"
echo ""
echo "# 5. List all available sections for your GPU"
echo "\$NCU --list-sections"