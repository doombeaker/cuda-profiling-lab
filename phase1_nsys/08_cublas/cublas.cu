#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>
#include "../../common/error_check.h"
#include "../../common/timer.h"

const int N = 1024;
const int K = 1024;
const int M = 1024;

int main() {
  // Step 1: Allocate host pinned memory
  float *A, *B, *C;
  CUDA_CHECK(cudaMallocHost(&A, N * K * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&B, K * M * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&C, N * M * sizeof(float)));

  // Step 2: Fill A and B with random floats, C with zeros
  for (int i = 0; i < N * K; i++) {
    A[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < K * M; i++) {
    B[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < N * M; i++) {
    C[i] = 0.0f;
  }

  // Step 3: Allocate device memory
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, N * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * M * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, N * M * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_C, 0, N * M * sizeof(float)));

  // Step 4: H2D copy A, B to device
  CUDA_CHECK(cudaMemcpy(d_A, A, N * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, K * M * sizeof(float), cudaMemcpyHostToDevice));

  // Step 5: Create cuBLAS handle
  cublasHandle_t handle;
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasCreate failed: %d\n", stat);
    std::exit(EXIT_FAILURE);
  }

  // Step 6: Warmup run to avoid first-launch overhead
  // Note: cuBLAS uses column-major layout. Our host data is row-major.
  // We swap A and B: C = B * A in column-major = (A * B)^T in column-major.
  // Reading back as row-major gives (A * B)^T^T = A * B.
  float alpha = 1.0f, beta = 0.0f;
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha, d_B, M, d_A, K,
                     &beta, d_C, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasSgemm warmup failed: %d\n", stat);
    std::exit(EXIT_FAILURE);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Step 7: Timed run
  GpuTimer timer;
  timer.start();
  stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha, d_B, M, d_A, K,
                     &beta, d_C, M);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cublasSgemm timed run failed: %d\n", stat);
    std::exit(EXIT_FAILURE);
  }
  timer.stop();

  // Step 8: Print timing
  printf("cuBLAS SGEMM (1024x1024): GPU time = %.3f ms\n", timer.elapsed_ms());

  // Step 9: D2H copy C back
  CUDA_CHECK(cudaMemcpy(C, d_C, N * M * sizeof(float), cudaMemcpyDeviceToHost));

  // Step 10: Spot-verify against CPU reference
  bool pass = true;
  for (int i = 0; i < 5; i++) {
    int row = rand() % N;
    int col = rand() % M;
    float expected = 0.0f;
    for (int k = 0; k < K; k++) {
      expected += A[row * K + k] * B[k * M + col];
    }
    if (fabsf(C[row * M + col] - expected) > 1e-1f) {
      pass = false;
      break;
    }
  }
  if (pass) {
    printf("Verification: PASS (spot-check)\n");
  } else {
    printf("Verification: FAIL\n");
  }

  // Step 11: Clean up
  cublasDestroy(handle);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  CUDA_CHECK(cudaFreeHost(A));
  CUDA_CHECK(cudaFreeHost(B));
  CUDA_CHECK(cudaFreeHost(C));

  return 0;
}