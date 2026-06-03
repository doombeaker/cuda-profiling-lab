#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cstdio>
#include <cstdlib>

// Kernel that reads AND writes — causes page faults on first GPU access
__global__ void kernel_scale(float *data, float scale, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

// Kernel that only writes — initializes pages on GPU without reading
__global__ void kernel_fill(float *data, float value, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    data[idx] = value;
  }
}

int main() {
  const int n = 1 << 24;  // 16M floats = 64 MB
  const size_t bytes = n * sizeof(float);
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;

  GpuTimer gpu_timer;
  CpuTimer cpu_timer;

  // ============================================================
  // Scenario 1: CPU-init → GPU-access (cold — page faults)
  // ============================================================
  printf("=== Scenario 1: CPU-init -> GPU-access (cold, page faults) ===\n");
  {
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    // Initialize on CPU — pages now resident on CPU
    for (int i = 0; i < n; i++) data[i] = 1.0f;

    // Launch kernel that reads AND writes.
    // GPU tries to read data[i] → page fault → UVM migrates page from CPU to GPU.
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms1 = gpu_timer.elapsed_ms();
    printf("  kernel time = %.3f ms\n", ms1);

    // Verify: 1.0 * 2.0 = 2.0
    CUDA_CHECK(cudaDeviceSynchronize());
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
      float diff = data[i] - 2.0f;
      if (diff < 0) diff = -diff;
      if (diff > max_err) max_err = diff;
    }
    printf("  max error = %.6f\n", max_err);

    CUDA_CHECK(cudaFree(data));
  }

  // ============================================================
  // Scenario 2: GPU-init → GPU-access (hot — no page faults)
  // ============================================================
  printf("\n=== Scenario 2: GPU-init -> GPU-access (hot, no faults) ===\n");
  {
    float *data2;
    CUDA_CHECK(cudaMallocManaged(&data2, bytes));

    // Initialize on GPU — pages now resident on GPU
    kernel_fill<<<blocks, threads>>>(data2, 3.0f, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch scale kernel — data already on GPU, no page faults
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data2, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms2 = gpu_timer.elapsed_ms();
    printf("  kernel time = %.3f ms\n", ms2);

    // Verify: 3.0 * 2.0 = 6.0
    CUDA_CHECK(cudaDeviceSynchronize());
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
      float diff = data2[i] - 6.0f;
      if (diff < 0) diff = -diff;
      if (diff > max_err) max_err = diff;
    }
    printf("  max error = %.6f\n", max_err);

    CUDA_CHECK(cudaFree(data2));
  }

  // ============================================================
  // Scenario 3: GPU-init → CPU-access (migration back to CPU)
  // ============================================================
  printf("\n=== Scenario 3: GPU-init -> CPU-access (migration back) ===\n");
  {
    float *data2;
    CUDA_CHECK(cudaMallocManaged(&data2, bytes));

    // Initialize and process on GPU (same as scenario 2)
    kernel_fill<<<blocks, threads>>>(data2, 3.0f, n);
    CUDA_CHECK_KERNEL();
    kernel_scale<<<blocks, threads>>>(data2, 2.0f, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    // Data is now resident on GPU

    // CPU reads all elements — forces migration back to CPU
    cpu_timer.start();
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
      sum += (double)data2[i];
    }
    cpu_timer.stop();
    double ms3 = cpu_timer.elapsed_ms();
    printf("  CPU read time = %.3f ms\n", ms3);
    printf("  sum = %.1f (expected %.1f)\n", sum, 6.0 * n);

    CUDA_CHECK(cudaFree(data2));
  }

  // ============================================================
  // Scenario 4: Prefetch hint (cudaMemPrefetchAsync)
  // ============================================================
  printf("\n=== Scenario 4: Prefetch (CPU->GPU hint) ===\n");
  {
    float *data3;
    CUDA_CHECK(cudaMallocManaged(&data3, bytes));

    // Initialize on CPU
    for (int i = 0; i < n; i++) data3[i] = 1.0f;

    // Hint: migrate pages to GPU BEFORE kernel launch
    int device_id = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(data3, bytes, device_id));

    // Launch kernel — pages already on GPU, no page faults during kernel
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data3, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms4 = gpu_timer.elapsed_ms();
    printf("  kernel time = %.3f ms\n", ms4);

    // Verify: 1.0 * 2.0 = 2.0
    CUDA_CHECK(cudaDeviceSynchronize());
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
      float diff = data3[i] - 2.0f;
      if (diff < 0) diff = -diff;
      if (diff > max_err) max_err = diff;
    }
    printf("  max error = %.6f\n", max_err);

    CUDA_CHECK(cudaFree(data3));
  }

  // ============================================================
  // Summary: re-run all 4 scenarios for clean comparison
  // ============================================================
  printf("\n=== Summary (64 MB, sm_90) ===\n");
  printf("%-40s %s\n", "Scenario", "Kernel Time");
  printf("%-40s %s\n", "----------------------------------------", "-----------");

  // Re-run Scenario 1
  {
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));
    for (int i = 0; i < n; i++) data[i] = 1.0f;
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms = gpu_timer.elapsed_ms();
    printf("%-40s %.3f ms\n", "1. CPU-init -> GPU (cold)", ms);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data));
  }

  // Re-run Scenario 2
  {
    float *data2;
    CUDA_CHECK(cudaMallocManaged(&data2, bytes));
    kernel_fill<<<blocks, threads>>>(data2, 3.0f, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data2, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms = gpu_timer.elapsed_ms();
    printf("%-40s %.3f ms\n", "2. GPU-init -> GPU (hot)", ms);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data2));
  }

  // Re-run Scenario 3 (CPU read)
  {
    float *data2;
    CUDA_CHECK(cudaMallocManaged(&data2, bytes));
    kernel_fill<<<blocks, threads>>>(data2, 3.0f, n);
    CUDA_CHECK_KERNEL();
    kernel_scale<<<blocks, threads>>>(data2, 2.0f, n);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaDeviceSynchronize());
    cpu_timer.start();
    volatile double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)data2[i];
    cpu_timer.stop();
    double ms = cpu_timer.elapsed_ms();
    printf("%-40s %.3f ms (CPU)\n", "3. GPU-init -> CPU read", ms);
    CUDA_CHECK(cudaFree(data2));
  }

  // Re-run Scenario 4
  {
    float *data3;
    CUDA_CHECK(cudaMallocManaged(&data3, bytes));
    for (int i = 0; i < n; i++) data3[i] = 1.0f;
    CUDA_CHECK(cudaMemPrefetchAsync(data3, bytes, 0));
    gpu_timer.start();
    kernel_scale<<<blocks, threads>>>(data3, 2.0f, n);
    CUDA_CHECK_KERNEL();
    gpu_timer.stop();
    float ms = gpu_timer.elapsed_ms();
    printf("%-40s %.3f ms\n", "4. Prefetch (CPU->GPU hint)", ms);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data3));
  }

  printf("\n");
  printf("Expected: Scenario 1 (cold) > Scenario 2 (hot) ≈ Scenario 4 (prefetch)\n");
  printf("Scenario 3 (CPU read) shows migration cost back to CPU.\n");
  printf("Open nsys-ui to see 'Unified Memory' row with page fault events.\n");

  return 0;
}