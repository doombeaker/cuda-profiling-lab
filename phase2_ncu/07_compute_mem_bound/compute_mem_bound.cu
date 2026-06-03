#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../../common/error_check.h"
#include "../../common/timer.h"

__global__ void kernel_memory_bound(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

__global__ void kernel_compute_bound(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float ai = a[i];
    float bi = b[i];
    float ci = 0.0f;
    for (int j = 0; j < 1024; j++) {
      ci = sinf(ai) * cosf(bi) + expf(ai * 0.001f) + logf(fabsf(bi) + 1.0f);
    }
    c[i] = ci;
  }
}

int main() {
  int N = 1 << 24;
  size_t bytes = N * sizeof(float);

  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);

  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(rand()) / RAND_MAX;
    h_b[i] = (float)(rand()) / RAND_MAX + 0.5f;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  int blockSize = 256;
  int gridSize = (N + blockSize - 1) / blockSize;

  GpuTimer timer1;
  timer1.start();
  kernel_memory_bound<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  timer1.stop();
  printf("memory-bound: %.3f ms\n", timer1.elapsed_ms());

  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();

  GpuTimer timer2;
  timer2.start();
  kernel_compute_bound<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  timer2.stop();
  printf("compute-bound: %.3f ms\n", timer2.elapsed_ms());

  printf("Arithmetic Intensity: memory-bound = 0.08 FLOP/byte, compute-bound = ~800 FLOP/byte\n");
  printf("Check ncu 'GPU Speed of Light' section — which kernel hits memory/compute limits?\n");

  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}
