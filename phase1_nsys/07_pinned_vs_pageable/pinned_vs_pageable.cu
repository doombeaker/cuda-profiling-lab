#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cstdlib>
#include <cstring>

__global__ void trivial_kernel(float *data, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    data[idx] += 1.0f;
  }
}

int main() {
  constexpr int TEST_SIZE = 1 << 26;  // 64M floats = 256 MB
  const int N = TEST_SIZE;
  const size_t BYTES = N * sizeof(float);
  const float SIZE_GB = (float)BYTES / (1024.0f * 1024.0f * 1024.0f);

  GpuTimer timer;

  // ============================================================
  // Test 1: Pinned memory (cudaMallocHost)
  // ============================================================
  {
    float *h_pinned, *d_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, BYTES));
    CUDA_CHECK(cudaMalloc(&d_pinned, BYTES));

    // Fill host buffer with a pattern
    std::memset(h_pinned, 0xAB, BYTES);

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_pinned, h_pinned, BYTES, cudaMemcpyHostToDevice));
    timer.stop();
    float pinned_ms = timer.elapsed_ms();
    float pinned_bw = SIZE_GB / (pinned_ms / 1000.0f);
    printf("Pinned (cudaMallocHost): H2D = %.3f ms (%.2f GB/s)\n",
           pinned_ms, pinned_bw);

    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFree(d_pinned));
  }

  // ============================================================
  // Test 2: Pageable memory (malloc)
  // ============================================================
  {
    float *h_pageable = (float *)std::malloc(BYTES);
    float *d_pageable;
    CUDA_CHECK(cudaMalloc(&d_pageable, BYTES));

    // Fill host buffer with same pattern
    std::memset(h_pageable, 0xAB, BYTES);

    timer.start();
    CUDA_CHECK(
        cudaMemcpy(d_pageable, h_pageable, BYTES, cudaMemcpyHostToDevice));
    timer.stop();
    float pageable_ms = timer.elapsed_ms();
    float pageable_bw = SIZE_GB / (pageable_ms / 1000.0f);
    printf("Pageable (malloc):      H2D = %.3f ms (%.2f GB/s)\n",
           pageable_ms, pageable_bw);

    // ============================================================
    // Test 3: Run a trivial kernel on the pageable destination
    //         to show the full lifecycle on the nsys timeline
    // ============================================================
    timer.start();
    trivial_kernel<<<(N + 255) / 256, 256>>>(d_pageable, N);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    float kernel_ms = timer.elapsed_ms();
    printf("  kernel on pageable data = %.3f ms\n", kernel_ms);

    std::free(h_pageable);
    CUDA_CHECK(cudaFree(d_pageable));
  }

  // ============================================================
  // Summary
  // ============================================================
  // Re-run pinned measurement for clean comparison (no warm-up bias)
  {
    float *h_pinned2, *d_pinned2;
    CUDA_CHECK(cudaMallocHost(&h_pinned2, BYTES));
    CUDA_CHECK(cudaMalloc(&d_pinned2, BYTES));
    std::memset(h_pinned2, 0xAB, BYTES);

    timer.start();
    CUDA_CHECK(
        cudaMemcpy(d_pinned2, h_pinned2, BYTES, cudaMemcpyHostToDevice));
    timer.stop();
    float pinned_ms2 = timer.elapsed_ms();
    float pinned_bw2 = SIZE_GB / (pinned_ms2 / 1000.0f);

    float *h_pageable2 = (float *)std::malloc(BYTES);
    float *d_pageable2;
    CUDA_CHECK(cudaMalloc(&d_pageable2, BYTES));
    std::memset(h_pageable2, 0xAB, BYTES);

    timer.start();
    CUDA_CHECK(
        cudaMemcpy(d_pageable2, h_pageable2, BYTES, cudaMemcpyHostToDevice));
    timer.stop();
    float pageable_ms2 = timer.elapsed_ms();
    float pageable_bw2 = SIZE_GB / (pageable_ms2 / 1000.0f);

    float speedup = pageable_ms2 / pinned_ms2;
    printf("\n=== Summary (256 MB H2D) ===\n");
    printf("Pinned:   %.3f ms (%.2f GB/s)\n", pinned_ms2, pinned_bw2);
    printf("Pageable: %.3f ms (%.2f GB/s)\n", pageable_ms2, pageable_bw2);
    printf("Speedup (pinned vs pageable): %.2fx\n", speedup);

    CUDA_CHECK(cudaFreeHost(h_pinned2));
    CUDA_CHECK(cudaFree(d_pinned2));
    std::free(h_pageable2);
    CUDA_CHECK(cudaFree(d_pageable2));
  }

  return 0;
}