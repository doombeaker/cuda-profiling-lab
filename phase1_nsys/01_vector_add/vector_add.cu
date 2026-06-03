#include "../../common/error_check.h"
#include "../../common/timer.h"

constexpr int VECTOR_SIZE = 1 << 26;
constexpr int BLOCK_SIZE = 256;

__global__ void vector_add(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  float *h_a, *h_b, *h_c;
  CUDA_CHECK(cudaMallocHost(&h_a, VECTOR_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_b, VECTOR_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_c, VECTOR_SIZE * sizeof(float)));

  for (int i = 0; i < VECTOR_SIZE; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, VECTOR_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, VECTOR_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, VECTOR_SIZE * sizeof(float)));

  GpuTimer timer;
  timer.start();

  CUDA_CHECK(cudaMemcpy(d_a, h_a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  int grid = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  vector_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, VECTOR_SIZE);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(h_c, d_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

  timer.stop();
  printf("Vector Add (N=%d): GPU time = %.3f ms\n", VECTOR_SIZE, timer.elapsed_ms());

  bool pass = true;
  int fail_idx = -1;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (h_c[i] != 3.0f) {
      pass = false;
      fail_idx = i;
      break;
    }
  }

  if (pass) {
    printf("Verification: PASS\n");
  } else {
    printf("Verification: FAIL at index %d (got %.1f, expected 3.0)\n", fail_idx, h_c[fail_idx]);
  }

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  CUDA_CHECK(cudaFreeHost(h_c));

  return 0;
}
