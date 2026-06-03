#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cstdio>

#define N (1 << 22)  // 4M floats = 16 MB, larger than L2 on most GPUs

// Kernel 1: l1_hit — repeated small working set that fits in L1 cache.
// Each thread reads one element, then processes it 128 times in registers.
// L1 hit rate should be high because each element is loaded once and re-used.
__global__ void l1_hit(float *data, float *output, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  float val = data[tid];
  for (int i = 0; i < 128; i++) {
    val = val * 0.999f + 0.001f;  // re-use the cached value
  }
  output[tid] = val;
}

// Kernel 2: l2_hit — medium working set, repeated access within L2 cache range.
// Each block processes a strip of 4096 elements repeatedly through shared memory.
// After the first pass, subsequent passes hit L2 cache.
__global__ void l2_hit(float *data, float *output, int n) {
  __shared__ float buf[256];
  int tid = threadIdx.x;
  int base = blockIdx.x * 256;
  if (base + tid >= n) return;
  float sum = 0.0f;
  // Each block processes 4096 elements in 16 passes through shared memory
  int strip_start = (blockIdx.x * 4096) % n;
  for (int pass = 0; pass < 16; pass++) {
    buf[tid] = data[(strip_start + pass * 256 + tid) % n];
    __syncthreads();
    for (int j = 0; j < 256; j++) sum += buf[j] * 0.1f;
    __syncthreads();
  }
  output[base + tid] = sum;
}

// Kernel 3: hbm_only — streaming access, no reuse, large array.
// Each element loaded exactly once. No L1/L2 reuse → mainly HBM traffic.
__global__ void hbm_only(float *data, float *output, int n, float scale, float bias) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= n) return;
  output[tid] = data[tid] * scale + bias;
}

int main() {
  size_t bytes = N * sizeof(float);

  // Use pageable memory (malloc) for clearer cache behavior
  float *h_data = (float *)malloc(bytes);
  float *h_output = (float *)malloc(bytes);
  float *d_data, *d_output;
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, bytes));

  // Initialize with random values
  for (int i = 0; i < N; i++) {
    h_data[i] = (float)rand() / (float)RAND_MAX;
  }

  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

  int blocks = (N + 255) / 256;

  // --- Kernel 1: l1_hit ---
  GpuTimer timer1;
  timer1.start();
  l1_hit<<<blocks, 256>>>(d_data, d_output, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  timer1.stop();
  printf("l1_hit: %.3f ms (high L1 reuse)\n", timer1.elapsed_ms());

  // --- Kernel 2: l2_hit ---
  GpuTimer timer2;
  timer2.start();
  l2_hit<<<N / 256, 256>>>(d_data, d_output, N);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  timer2.stop();
  printf("l2_hit: %.3f ms (medium L2 reuse)\n", timer2.elapsed_ms());

  // --- Kernel 3: hbm_only ---
  GpuTimer timer3;
  timer3.start();
  hbm_only<<<blocks, 256>>>(d_data, d_output, N, 2.0f, 1.0f);
  CUDA_CHECK_KERNEL();
  cudaDeviceSynchronize();
  timer3.stop();
  printf("hbm_only: %.3f ms (streaming, HBM-bound)\n", timer3.elapsed_ms());

  cudaFree(d_data);
  cudaFree(d_output);
  free(h_data);
  free(h_output);

  return 0;
}