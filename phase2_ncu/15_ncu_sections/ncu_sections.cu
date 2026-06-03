#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>

// GEMV with shared memory tiling.
// Exercises: global memory (A, x reads), shared memory (x_tile),
// __syncthreads(), FP32 compute, loop control flow.
__global__ void gemv_tiled(float *A, float *x, float *y, int rows, int cols) {
    extern __shared__ float x_tile[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < cols; t += blockDim.x) {
        int idx = t + tid;
        x_tile[tid] = (idx < cols) ? x[idx] : 0.0f;
        __syncthreads();
        for (int j = 0; j < blockDim.x && (t + j) < cols; j++) {
            sum += A[row * cols + t + j] * x_tile[j];
        }
        __syncthreads();
    }
    if (tid == 0) y[row] = sum;
}

int main() {
    const int rows = 4096;
    const int cols = 4096;
    const int blockSize = 256;

    size_t bytes_A = rows * cols * sizeof(float);
    size_t bytes_x = cols * sizeof(float);
    size_t bytes_y = rows * sizeof(float);

    // Pinned host memory
    float *h_A, *h_x, *h_y;
    CUDA_CHECK(cudaMallocHost(&h_A, bytes_A));
    CUDA_CHECK(cudaMallocHost(&h_x, bytes_x));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes_y));

    // Fill A and x with random values
    for (int i = 0; i < rows * cols; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < cols; i++) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device memory
    float *d_A, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_x, bytes_x));
    CUDA_CHECK(cudaMalloc(&d_y, bytes_y));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice));

    // Launch: one block per row, dynamic shared memory for x tile
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer;
    timer.start();
    gemv_tiled<<<rows, blockSize, blockSize * sizeof(float)>>>(d_A, d_x, d_y, rows, cols);
    CUDA_CHECK_KERNEL();
    timer.stop();
    float ms = timer.elapsed_ms();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes_y, cudaMemcpyDeviceToHost));

    std::printf("GEMV tiled: rows=%d cols=%d blockSize=%d\n", rows, cols, blockSize);
    std::printf("Kernel time: %.3f ms\n", ms);

    // Verify one row (row 0) for correctness
    float expected = 0.0f;
    for (int j = 0; j < cols; j++) {
        expected += h_A[0 * cols + j] * h_x[j];
    }
    bool pass = std::fabs(h_y[0] - expected) < 1e-3f;
    std::printf("Row 0 verification: %s (expected=%.6f, got=%.6f)\n",
                pass ? "PASS" : "FAIL", expected, h_y[0]);

    // Print profiling variations to try
    std::printf("\n=== Try these ncu profiling variations ===\n");
    std::printf("NCU=/usr/local/cuda-12.4/nsight-compute-2024.1.1/ncu\n");
    std::printf("EXE=./ncu_sections\n\n");
    std::printf("# 1. --set analysis (fast, essential metrics)\n");
    std::printf("$NCU --set analysis --print-summary $EXE\n\n");
    std::printf("# 2. --section SpeedOfLight (fastest, high-level bottleneck)\n");
    std::printf("$NCU --section SpeedOfLight --print-summary $EXE\n\n");
    std::printf("# 3. --section MemoryWorkloadAnalysis (targeted at memory)\n");
    std::printf("$NCU --section MemoryWorkloadAnalysis --print-summary $EXE\n\n");
    std::printf("# 4. --section Occupancy (launch config tuning)\n");
    std::printf("$NCU --section Occupancy --print-summary $EXE\n\n");
    std::printf("# 5. Compare --set analysis vs --set full timing\n");
    std::printf("time $NCU --set analysis $EXE\n");
    std::printf("time $NCU --set full $EXE\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));

    return 0;
}