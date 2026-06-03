#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../../common/error_check.h"
#include "../../common/timer.h"

// Kernel 1: Pure FP32 arithmetic — dominated by FP32 instructions
__global__ void fp32_only(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = a[i];
  // Pure FP32 operations (fma, mul, add, sin, cos)
  v = sinf(v) * 0.5f + cosf(v) * 0.3f;
  v = v * v + v * 0.1f;
  v = fmaf(v, 2.0f, 1.0f);
  c[i] = v + b[i];
}

// Kernel 2: Dominated by integer arithmetic with some FP32 wrapping
__global__ void int_heavy(float *a, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float val = a[i];
  // Convert to int, do heavy integer computation, convert back
  int ival = (int)(val * 1000.0f);
  int result = 0;
  for (int k = 0; k < 200; k++) {
    ival = ival * 1103515245 + 12345; // linear congruential generator
    result ^= ival;
    result = (result << 13) | (result >> 19); // bit rotation
    result += ival % 997;
  }
  c[i] = (float)(result & 0xFFFFF) / 1048576.0f;
}

// Kernel 3: Lots of branching and control flow
__global__ void control_heavy(float *a, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float val = a[i];
  float result = 0.0f;
  // Many branches based on the value
  for (int k = 0; k < 200; k++) {
    if (val < 0.1f) {
      result += val * 10.0f;
    } else if (val < 0.3f) {
      result += val * 5.0f;
    } else if (val < 0.5f) {
      result += val * 2.0f;
    } else if (val < 0.7f) {
      result -= val * 0.5f;
    } else if (val < 0.9f) {
      result -= val * 2.0f;
    } else {
      result -= val * 5.0f;
    }
    val = val * 1.01f - floorf(val * 1.01f); // shift value
    if (k % 10 == 0) result *= 0.99f;
  }
  c[i] = result;
}

int main() {
  int N = 1 << 22; // 4M elements
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

  // Kernel 1: fp32_only
  GpuTimer timer1;
  timer1.start();
  fp32_only<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  timer1.stop();
  printf("fp32_only: %.3f ms (dominated by FP32 instructions)\n", timer1.elapsed_ms());

  // Kernel 2: int_heavy
  GpuTimer timer2;
  timer2.start();
  int_heavy<<<gridSize, blockSize>>>(d_a, d_c, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  timer2.stop();
  printf("int_heavy: %.3f ms (dominated by INT instructions)\n", timer2.elapsed_ms());

  // Kernel 3: control_heavy
  GpuTimer timer3;
  timer3.start();
  control_heavy<<<gridSize, blockSize>>>(d_a, d_c, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  cudaDeviceSynchronize();
  timer3.stop();
  printf("control_heavy: %.3f ms (dominated by control/logic instructions)\n", timer3.elapsed_ms());

  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}