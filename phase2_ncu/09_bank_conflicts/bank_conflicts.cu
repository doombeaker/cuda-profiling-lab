#include "../../common/error_check.h"
#include "../../common/timer.h"

#define N (1 << 24)  // 16M elements

// Kernel 1: stride-1 access — each thread accesses its own bank, no conflict
__global__ void no_conflict(float *input, float *output, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    output[i] = smem[tid] * 2.0f;  // stride-1, no conflict
}

// Kernel 2: stride-2 access — every 16 pairs of threads share banks (2-way conflict)
__global__ void conflict_2way(float *input, float *output, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    int idx2 = (tid * 2) % 256;
    smem[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    output[i] = smem[idx2] * 2.0f;  // stride-2 read, 2-way conflict
}

// Kernel 3: stride-32 access — all 32 threads fight for same bank (32-way conflict, worst case)
__global__ void conflict_32way(float *input, float *output, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    output[i] = smem[0] * 2.0f;  // all threads read bank 0, 32-way conflict
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_input  = (float *)malloc(bytes);

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // Initialize input with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize  = N / blockSize;

    // --- Kernel 1: no_conflict ---
    GpuTimer timer1;
    timer1.start();
    no_conflict<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();
    timer1.stop();
    float t1 = timer1.elapsed_ms();

    // --- Kernel 2: conflict_2way ---
    GpuTimer timer2;
    timer2.start();
    conflict_2way<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();
    timer2.stop();
    float t2 = timer2.elapsed_ms();

    // --- Kernel 3: conflict_32way ---
    GpuTimer timer3;
    timer3.start();
    conflict_32way<<<gridSize, blockSize>>>(d_input, d_output, N);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();
    timer3.stop();
    float t3 = timer3.elapsed_ms();

    // --- Print timings ---
    printf("no_conflict:    %.3f ms (fastest)\n", t1);
    printf("conflict_2way:  %.3f ms (slower)\n", t2);
    printf("conflict_32way: %.3f ms (slowest)\n", t3);

    // --- Cleanup ---
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return 0;
}