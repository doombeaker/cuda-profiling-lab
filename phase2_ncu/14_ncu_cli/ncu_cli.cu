#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>

// Kernel 1: SAXPY — y[i] = a * x[i] + y[i] (classic memory-bound BLAS-1)
__global__ void saxpy(float *x, float *y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

// Kernel 2: dot — partial dot product per block (compute + reduction)
__global__ void dot_partial(float *x, float *y, float *partial, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = (i < n) ? x[i] * y[i] : 0.0f;
    sdata[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Kernel 3: scale — simple element-wise scale
__global__ void scale(float *x, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= factor;
}

int main() {
    const int N = 1 << 22;  // 4M floats
    size_t bytes = N * sizeof(float);

    // Allocate pinned host memory
    float *h_x, *h_y;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));

    // Fill with test data
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_partial;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    int blocks = (N + 255) / 256;
    int threads = 256;
    int numBlocks = blocks;
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(float)));

    // H2D copy
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // --- Kernel 1: saxpy ---
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer_saxpy;
    timer_saxpy.start();
    saxpy<<<blocks, threads>>>(d_x, d_y, 2.0f, N);
    CUDA_CHECK_KERNEL();
    timer_saxpy.stop();
    float ms_saxpy = timer_saxpy.elapsed_ms();
    std::printf("saxpy: %.3f ms\n", ms_saxpy);

    // --- Kernel 2: dot_partial ---
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer_dot;
    timer_dot.start();
    dot_partial<<<blocks, threads>>>(d_x, d_y, d_partial, N);
    CUDA_CHECK_KERNEL();
    timer_dot.stop();
    float ms_dot = timer_dot.elapsed_ms();

    // D2H partial results and CPU sum
    float *h_partial = (float *)malloc(numBlocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float dot_result = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        dot_result += h_partial[i];
    }
    std::printf("dot: %.3f ms, result=%.1f\n", ms_dot, dot_result);
    free(h_partial);

    // --- Kernel 3: scale ---
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer_scale;
    timer_scale.start();
    scale<<<blocks, threads>>>(d_y, 0.5f, N);
    CUDA_CHECK_KERNEL();
    timer_scale.stop();
    float ms_scale = timer_scale.elapsed_ms();
    std::printf("scale: %.3f ms\n", ms_scale);

    // --- Verification ---
    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    // After saxpy: y[i] = 2.0 * 1.0 + 2.0 = 4.0
    // After scale: y[i] = 4.0 * 0.5 = 2.0
    float expected = 2.0f;
    bool pass = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_y[i] - expected) > 1e-3f) {
            pass = false;
            std::printf("FAIL at index %d: expected %.1f, got %.3f\n",
                        i, expected, h_y[i]);
            break;
        }
    }
    std::printf("Verification: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_partial));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));

    return 0;
}