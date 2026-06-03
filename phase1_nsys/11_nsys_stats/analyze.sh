#!/bin/bash
set -e
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
REPORT=./report11.nsys-rep

echo "=== Kernel Execution Summary ==="
$NSYS stats --report cuda_gpu_kern_sum $REPORT

echo ""
echo "=== Memory Operation Summary ==="
$NSYS stats --report cuda_gpu_mem_size_sum $REPORT

echo ""
echo "=== CUDA API Summary ==="
$NSYS stats --report cuda_api_sum $REPORT

echo ""
echo "Done. Add --format csv for machine-readable output."