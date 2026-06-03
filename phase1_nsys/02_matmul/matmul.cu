#include <cstdio>
#include <cstdlib>
#include "../../common/error_check.h"
#include "../../common/timer.h"

const int N = 1024;
const int TILE = 16;

__global__ void matmul_naive(float *A, float *B, float *C, int N) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < N; k++) {
      sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

__global__ void matmul_tiled(float *A, float *B, float *C, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float sum = 0.0f;
  for (int t = 0; t < N / TILE; t++) {
    As[threadIdx.y][threadIdx.x] = A[row * N + (t * TILE + threadIdx.x)];
    Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
    __syncthreads();
    for (int k = 0; k < TILE; k++) {
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    __syncthreads();
  }
  C[row * N + col] = sum;
}

int main() {
  size_t bytes = N * N * sizeof(float);

  float *A, *B, *C;
  CUDA_CHECK(cudaMallocHost(&A, bytes));
  CUDA_CHECK(cudaMallocHost(&B, bytes));
  CUDA_CHECK(cudaMallocHost(&C, bytes));

  for (int i = 0; i < N * N; i++) {
    A[i] = rand() / (float)RAND_MAX;
    B[i] = rand() / (float)RAND_MAX;
  }

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  CUDA_CHECK(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

  GpuTimer timer_naive;
  timer_naive.start();
  matmul_naive<<<dim3(N / 16, N / 16), dim3(16, 16)>>>(d_A, d_B, d_C, N);
  CUDA_CHECK_KERNEL();
  timer_naive.stop();
  printf("MatMul Naive (N=1024): GPU time = %.3f ms\n", timer_naive.elapsed_ms());

  CUDA_CHECK(cudaMemset(d_C, 0, bytes));

  GpuTimer timer_tiled;
  timer_tiled.start();
  matmul_tiled<<<dim3(N / TILE, N / TILE), dim3(TILE, TILE)>>>(d_A, d_B, d_C, N);
  CUDA_CHECK_KERNEL();
  timer_tiled.stop();
  printf("MatMul Tiled (N=1024): GPU time = %.3f ms\n", timer_tiled.elapsed_ms());

  CUDA_CHECK(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

  bool pass = true;
  for (int i = 0; i < 5; i++) {
    int idx = rand() % (N * N);
    int row = idx / N, col = idx % N;
    float expected = 0.0f;
    for (int k = 0; k < N; k++) {
      expected += A[row * N + k] * B[k * N + col];
    }
    if (fabsf(C[idx] - expected) > 1e-1f) {
      pass = false;
      break;
    }
  }
  if (pass) {
    printf("Verification: done (spot-check passed)\n");
  }

  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFreeHost(A));
  CUDA_CHECK(cudaFreeHost(B));
  CUDA_CHECK(cudaFreeHost(C));

  return 0;
}
