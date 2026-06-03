#include "../../common/error_check.h"
#include "../../common/timer.h"

constexpr int N_ELEMS = 1024;
constexpr int BLOCK_SIZE = 256;

// Trivial empty kernel — used to measure pure launch overhead
__global__ void noop_kernel() {}

// Simple addition kernel — used for comparison
__global__ void tiny_add(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  // Allocate device memory for the work kernel
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N_ELEMS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N_ELEMS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_c, N_ELEMS * sizeof(float)));

  // Initialize with a simple memset-like kernel or host copy
  float *h_init = new float[N_ELEMS];
  for (int i = 0; i < N_ELEMS; i++) {
    h_init[i] = static_cast<float>(i);
  }
  CUDA_CHECK(cudaMemcpy(d_a, h_init, N_ELEMS * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_init, N_ELEMS * sizeof(float), cudaMemcpyHostToDevice));
  delete[] h_init;

  GpuTimer timer;

  printf("Kernel Launch Overhead Test\n");
  printf("===========================\n\n");

  // ── Experiment 1: NOOP kernel with varying grid sizes ──
  printf("noop_kernel (1 thread/block):\n");

  int grid_sizes[] = {1, 10, 100, 1000};
  for (int g : grid_sizes) {
    timer.start();
    noop_kernel<<<g, 1>>>();
    CUDA_CHECK_KERNEL();
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  %4d blocks:  %.3f ms\n", g, timer.elapsed_ms());
  }

  printf("\n");

  // ── Experiment 2: TINY_ADD single launch ──
  printf("tiny_add (%d threads, %d elems):\n", BLOCK_SIZE, N_ELEMS);

  int grid = (N_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  timer.start();
  tiny_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N_ELEMS);
  CUDA_CHECK_KERNEL();
  timer.stop();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("  1 launch:   %.3f ms\n", timer.elapsed_ms());

  // ── Experiment 3: TINY_ADD 100 sequential launches ──
  timer.start();
  for (int i = 0; i < 100; i++) {
    tiny_add<<<grid, BLOCK_SIZE>>>(d_a, d_b, d_c, N_ELEMS);
    CUDA_CHECK_KERNEL();
  }
  timer.stop();
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("  100 launches: %.3f ms\n", timer.elapsed_ms());

  printf("\n");

  // ── Verification ──
  float *h_c = new float[N_ELEMS];
  CUDA_CHECK(cudaMemcpy(h_c, d_c, N_ELEMS * sizeof(float), cudaMemcpyDeviceToHost));

  bool pass = true;
  for (int i = 0; i < N_ELEMS; i++) {
    if (h_c[i] != static_cast<float>(i + i)) {
      pass = false;
      printf("Verification: FAIL at index %d (got %.1f, expected %.1f)\n",
             i, h_c[i], static_cast<float>(i + i));
      break;
    }
  }
  if (pass) {
    printf("Verification: PASS (no CUDA errors)\n");
  }

  delete[] h_c;
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}