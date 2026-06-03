#include "../../common/error_check.h"
#include "../../common/timer.h"

__global__ void trivial_kernel(float *data, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    data[idx] += 1.0f;
  }
}

int main() {
  const int SIZES[] = {1 << 20, 1 << 22, 1 << 24, 1 << 26, 1 << 28};
  const int NUM_SIZES = 5;

  GpuTimer timer;

  for (int i = 0; i < NUM_SIZES; i++) {
    int s = SIZES[i];
    float *h_data, *d_data;

    CUDA_CHECK(cudaMallocHost(&h_data, s * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_data, s * sizeof(float)));

    for (int j = 0; j < s; j++) {
      h_data[j] = 0.0f;
    }

    float size_mb = (float)(s * sizeof(float)) / (1024.0f * 1024.0f);
    float size_gb = (float)(s * sizeof(float)) / (1024.0f * 1024.0f * 1024.0f);

    timer.start();
    CUDA_CHECK(cudaMemcpy(d_data, h_data, s * sizeof(float), cudaMemcpyHostToDevice));
    timer.stop();
    float h2d_ms = timer.elapsed_ms();
    float h2d_bw = size_gb / (h2d_ms / 1000.0f);
    printf("size=%.0f MB: H2D = %.3f ms (%.2f GB/s)\n", size_mb, h2d_ms, h2d_bw);

    timer.start();
    trivial_kernel<<<(s + 255) / 256, 256>>>(d_data, s);
    CUDA_CHECK(cudaGetLastError());
    timer.stop();
    float kern_ms = timer.elapsed_ms();
    printf("  kernel = %.3f ms\n", kern_ms);

    timer.start();
    CUDA_CHECK(cudaMemcpy(h_data, d_data, s * sizeof(float), cudaMemcpyDeviceToHost));
    timer.stop();
    float d2h_ms = timer.elapsed_ms();
    float d2h_bw = size_gb / (d2h_ms / 1000.0f);
    printf("  D2H   = %.3f ms (%.2f GB/s)\n", d2h_ms, d2h_bw);

    CUDA_CHECK(cudaFreeHost(h_data));
    CUDA_CHECK(cudaFree(d_data));
  }

  printf("\nCompare transfer times vs kernel time — is this workload memory-bound?\n");

  return 0;
}
