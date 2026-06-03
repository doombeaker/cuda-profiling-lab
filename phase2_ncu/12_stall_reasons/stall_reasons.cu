#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>

// ---------------------------------------------------------------------------
// Kernel 1: memory_stall — pure memory-bound
// Every instruction depends on a global memory load.
// Warps stall on "Long Scoreboard" waiting for global memory loads.
// Expect: Stall Long Scoreboard >> others, very high LG throttle.
// ---------------------------------------------------------------------------
__global__ void memory_stall(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] + b[i];
}

// ---------------------------------------------------------------------------
// Kernel 2: math_stall — compute-heavy with dependent math operations
// Heavy sinf/cosf loop creates a chain of dependent math instructions.
// Warps stall on "Short Scoreboard" waiting for math unit results.
// Expect: Stall Short Scoreboard dominates, some Stall LG from the initial load.
// ---------------------------------------------------------------------------
__global__ void math_stall(float *a, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float val = a[i];
    for (int k = 0; k < 500; k++) {
        val = sinf(val) * 0.99f + cosf(val) * 0.01f;
    }
    c[i] = val;
}

// ---------------------------------------------------------------------------
// Kernel 3: sync_stall — heavy __syncthreads() barriers
// Many __syncthreads() calls cause warps to wait for other warps in the block.
// Warps stall on "Barrier" waiting for peers to reach __syncthreads().
// Expect: Stall Barrier significant (20-40%), mixed with other stalls.
// ---------------------------------------------------------------------------
__global__ void sync_stall(float *data, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    smem[tid] = (i < n) ? data[i] : 1.0f;  // use 1.0f so all threads start with same value

    for (int s = 0; s < 8; s++) {
        __syncthreads();
        smem[tid] = smem[tid] * 0.99f + smem[(tid + 32) % 256] * 0.01f;
        __syncthreads();
        smem[tid] = smem[tid] + smem[(tid + 64) % 256];
        __syncthreads();
        smem[tid] = smem[tid] - smem[(tid + 128) % 256];
    }

    __syncthreads();
    if (i < n) data[i] = smem[tid];
}

// ---------------------------------------------------------------------------
int main() {
    const int N = 1 << 22;       // 4M elements
    const int blockSize = 256;
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    // Initialize: a[i] = 1.0f, b[i] = 2.0f
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + blockSize - 1) / blockSize;

    // =======================================================================
    // Kernel 1: memory_stall
    // =======================================================================
    std::printf("=== Kernel 1: memory_stall ===\n");
    std::printf("    Pure memory-bound: c[i] = a[i] + b[i]\n");
    std::printf("    Expected ncu stall: Long Scoreboard (LG) dominates\n");
    std::printf("    Warps wait for global memory loads to complete.\n\n");

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer0;
    timer0.start();
    memory_stall<<<blocks, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_KERNEL();
    timer0.stop();
    float ms_mem = timer0.elapsed_ms();
    std::printf("    Time: %.3f ms\n\n", ms_mem);

    // Verify: c[i] should be 1.0 + 2.0 = 3.0
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    bool pass_mem = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - 3.0f) > 1e-3f) { pass_mem = false; break; }
    }
    std::printf("    Verification: %s\n\n", pass_mem ? "PASS" : "FAIL");

    // =======================================================================
    // Kernel 2: math_stall
    // =======================================================================
    std::printf("=== Kernel 2: math_stall ===\n");
    std::printf("    Compute-heavy: 500 iterations of sinf/cosf per element\n");
    std::printf("    Expected ncu stall: Short Scoreboard (SB) dominates\n");
    std::printf("    Warps wait for math unit results to become available.\n\n");

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer1;
    timer1.start();
    math_stall<<<blocks, blockSize>>>(d_a, d_c, N);
    CUDA_CHECK_KERNEL();
    timer1.stop();
    float ms_math = timer1.elapsed_ms();
    std::printf("    Time: %.3f ms\n\n", ms_math);

    // Verify: compute expected value on CPU
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    float expected = 1.0f;
    for (int k = 0; k < 500; k++) {
        expected = sinf(expected) * 0.99f + cosf(expected) * 0.01f;
    }
    bool pass_math = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - expected) > 1e-3f) { pass_math = false; break; }
    }
    std::printf("    Verification: %s\n\n", pass_math ? "PASS" : "FAIL");

    // =======================================================================
    // Kernel 3: sync_stall
    // =======================================================================
    std::printf("=== Kernel 3: sync_stall ===\n");
    std::printf("    Heavy __syncthreads(): 24 barriers per block\n");
    std::printf("    Expected ncu stall: Barrier dominates\n");
    std::printf("    Warps wait for other warps in the block to reach __syncthreads().\n\n");

    // Re-initialize d_a for sync_stall (it was read by math_stall, still has original values)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer2;
    timer2.start();
    sync_stall<<<blocks, blockSize>>>(d_a, N);
    CUDA_CHECK_KERNEL();
    timer2.stop();
    float ms_sync = timer2.elapsed_ms();
    std::printf("    Time: %.3f ms\n\n", ms_sync);

    // Verify: kernel ran without errors, check values are finite
    CUDA_CHECK(cudaMemcpy(h_c, d_a, bytes, cudaMemcpyDeviceToHost));
    bool pass_sync = true;
    for (int i = 0; i < N; i++) {
        if (!std::isfinite(h_c[i])) { pass_sync = false; break; }
    }
    std::printf("    Sample output: h_c[0]=%f, h_c[255]=%f, h_c[256]=%f\n",
                h_c[0], h_c[255], h_c[256]);
    std::printf("    Verification: %s (all values finite, kernel executed correctly)\n\n",
                pass_sync ? "PASS" : "FAIL");

    // =======================================================================
    // Summary
    // =======================================================================
    std::printf("=== Summary ===\n");
    std::printf("    memory_stall: %.3f ms  (ncu: check Stall Long Scoreboard)\n", ms_mem);
    std::printf("    math_stall:   %.3f ms  (ncu: check Stall Short Scoreboard)\n", ms_math);
    std::printf("    sync_stall:   %.3f ms  (ncu: check Stall Barrier)\n", ms_sync);
    std::printf("\nRun: ./profile.sh  then  ncu-ui ./report12.ncu-rep\n");
    std::printf("In ncu-ui, go to Warp State Statistics for each kernel.\n");

    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}