#include "../../common/error_check.h"
#include "../../common/timer.h"

#define N (1 << 24)
#define STRIDE 32

__global__ void kernel_coalesced(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = data[i] * 2.0f + 1.0f;
    }
}

__global__ void kernel_strided(float *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int idx = (i * STRIDE) % n;
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float *h_data = (float *)malloc(bytes);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)rand() / (float)RAND_MAX;
    }

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + 255) / 256;

    GpuTimer timer1;
    timer1.start();
    kernel_coalesced<<<blocks, 256>>>(d_data, N);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();
    timer1.stop();
    printf("coalesced: %.3f ms\n", timer1.elapsed_ms());

    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer2;
    timer2.start();
    kernel_strided<<<blocks, 256>>>(d_data, N);
    CUDA_CHECK_KERNEL();
    cudaDeviceSynchronize();
    timer2.stop();
    printf("strided: %.3f ms\n", timer2.elapsed_ms());

    printf("Bandwidth ratio: coalesced is %.1fx faster (check ncu for L1/L2 hit rates)\n",
           timer2.elapsed_ms() / timer1.elapsed_ms());

    cudaFree(d_data);
    free(h_data);

    return 0;
}
