#include "../../common/error_check.h"
#include "../../common/timer.h"

constexpr int N = 1 << 24;       // 16M floats ≈ 64 MB
constexpr int BLOCK_SIZE = 256;

// Kernel 1: element-wise vector add  c[i] = a[i] + b[i]
__global__ void vector_add(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// Kernel 2: element-wise scale  data[i] *= scalar
__global__ void vector_scale(float *data, float scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] *= scalar;
  }
}

// Kernel 3: SAXPY  b[i] = s * a[i] + b[i]
__global__ void vector_saxpy(float *a, float *b, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    b[i] = s * a[i] + b[i];
  }
}

int main() {
  const size_t bytes = N * sizeof(float);
  int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate pinned host memory
  float *h_a, *h_b, *h_c;
  CUDA_CHECK(cudaMallocHost(&h_a, bytes));
  CUDA_CHECK(cudaMallocHost(&h_b, bytes));
  CUDA_CHECK(cudaMallocHost(&h_c, bytes));

  // Initialize: a=1.0, b=2.0, c=0.0
  for (int i = 0; i < N; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
    h_c[i] = 0.0f;
  }

  // Allocate device memory
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  GpuTimer timer_h2d, timer_add, timer_scale, timer_saxpy, timer_d2h;

  // Step 1: H2D copy a, b to device
  timer_h2d.start();
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
  timer_h2d.stop();

  // Step 2: vector_add — fills c with 3.0f
  timer_add.start();
  vector_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
  CUDA_CHECK_KERNEL();
  timer_add.stop();

  // Step 3: vector_scale — scales c to 1.5f
  timer_scale.start();
  vector_scale<<<grid, BLOCK_SIZE>>>(d_c, 0.5f, N);
  CUDA_CHECK_KERNEL();
  timer_scale.stop();

  // Step 4: vector_saxpy — b becomes a+b = 3.0f
  timer_saxpy.start();
  vector_saxpy<<<grid, BLOCK_SIZE>>>(d_a, d_b, 1.0f, N);
  CUDA_CHECK_KERNEL();
  timer_saxpy.stop();

  // Step 5: D2H copy c back to host
  timer_d2h.start();
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
  timer_d2h.stop();

  // Print individual timings
  printf("N = %d (%.1f MB per array)\n", N, bytes / (1024.0f * 1024.0f));
  printf("--- Kernel Timings ---\n");
  printf("vector_add:   %.3f ms\n", timer_add.elapsed_ms());
  printf("vector_scale: %.3f ms\n", timer_scale.elapsed_ms());
  printf("vector_saxpy: %.3f ms\n", timer_saxpy.elapsed_ms());
  printf("--- Memory Transfer Timings ---\n");
  printf("H2D (a+b):    %.3f ms\n", timer_h2d.elapsed_ms());
  printf("D2H (c):      %.3f ms\n", timer_d2h.elapsed_ms());

  // Verify: c should be all 1.5f
  bool pass = true;
  int fail_idx = -1;
  for (int i = 0; i < N; i++) {
    if (h_c[i] != 1.5f) {
      pass = false;
      fail_idx = i;
      break;
    }
  }

  if (pass) {
    printf("Verification: PASS\n");
  } else {
    printf("Verification: FAIL at index %d (got %.1f, expected 1.5)\n",
           fail_idx, h_c[fail_idx]);
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFreeHost(h_a));
  CUDA_CHECK(cudaFreeHost(h_b));
  CUDA_CHECK(cudaFreeHost(h_c));

  return 0;
}