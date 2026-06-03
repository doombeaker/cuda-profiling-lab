#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// GEMV kernel: y = A * x  where A is N×N, x and y are N-element vectors.
// Each block computes one row of the output.
// Dynamic shared memory layout:
//   smem[0 .. N-1]        = x_shared (cached copy of x)
//   smem[N .. N+blockDim.x-1] = partial (reduction workspace)
__global__ void gemv_kernel(float *A, float *x, float *y, int N) {
  extern __shared__ float smem[];
  float *x_shared = smem;
  float *partial  = &smem[N];

  int row = blockIdx.x;
  int tid = threadIdx.x;
  float sum = 0.0f;

  // Cooperatively load the entire x vector into shared memory
  for (int k = tid; k < N; k += blockDim.x) {
    x_shared[k] = x[k];
  }
  __syncthreads();

  // Each thread computes its partial dot product (strided by blockDim.x)
  for (int k = tid; k < N; k += blockDim.x) {
    sum += A[row * N + k] * x_shared[k];
  }

  // Block-level reduction via shared memory
  partial[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) partial[tid] += partial[tid + s];
    __syncthreads();
  }
  if (tid == 0) y[row] = partial[0];
}

int main() {
  const int N = 2048;
  const int blockSizes[] = {32, 64, 128, 256, 512};
  const int numConfigs = sizeof(blockSizes) / sizeof(blockSizes[0]);

  size_t bytesA = (size_t)N * N * sizeof(float);
  size_t bytesV = (size_t)N * sizeof(float);

  // Allocate host memory
  float *h_A = (float *)malloc(bytesA);
  float *h_x = (float *)malloc(bytesV);
  float *h_y = (float *)malloc(bytesV);

  // Fill A with random floats in [1.0, 2.0), x with 1.0f
  for (int i = 0; i < N * N; i++) {
    h_A[i] = 1.0f + (float)rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < N; i++) {
    h_x[i] = 1.0f;
  }

  // Allocate device memory
  float *d_A, *d_x, *d_y;
  CUDA_CHECK(cudaMalloc(&d_A, bytesA));
  CUDA_CHECK(cudaMalloc(&d_x, bytesV));
  CUDA_CHECK(cudaMalloc(&d_y, bytesV));

  // Copy inputs to device
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytesV, cudaMemcpyHostToDevice));

  float bestMs = 1e9f;
  int bestBlockSize = 0;

  std::printf("=== GEMV Block Size Tuning (N=%d) ===\n\n", N);

  for (int i = 0; i < numConfigs; i++) {
    int blockSize = blockSizes[i];
    // Dynamic shared memory: N floats for x_shared + blockSize floats for partial
    size_t smemBytes = (N + blockSize) * sizeof(float);

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer;
    timer.start();
    gemv_kernel<<<N, blockSize, smemBytes>>>(d_A, d_x, d_y, N);
    CUDA_CHECK_KERNEL();
    timer.stop();
    float ms = timer.elapsed_ms();

    std::printf("blockSize=%3d: %.3f ms\n", blockSize, ms);

    if (ms < bestMs) {
      bestMs = ms;
      bestBlockSize = blockSize;
    }
  }

  std::printf("\nOptimal block size: %d (%.3f ms)\n", bestBlockSize, bestMs);

  // Verify row 0: expected = sum of row 0 of A (since x[i] = 1.0f)
  CUDA_CHECK(cudaMemcpy(h_y, d_y, bytesV, cudaMemcpyDeviceToHost));
  float expected = 0.0f;
  for (int k = 0; k < N; k++) {
    expected += h_A[0 * N + k];
  }
  bool pass = std::fabs(h_y[0] - expected) < 1e-2f;
  std::printf("Verification (row 0): %s (expected=%.3f, got=%.3f)\n",
              pass ? "PASS" : "FAIL", expected, h_y[0]);

  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  free(h_A);
  free(h_x);
  free(h_y);

  return pass ? 0 : 1;
}