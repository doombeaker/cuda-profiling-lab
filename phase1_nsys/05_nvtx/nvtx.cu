#include "../../common/error_check.h"
#include "../../common/timer.h"
#include "nvtx3/nvToolsExt.h"

constexpr int VECTOR_SIZE = 1 << 26;
constexpr int BLOCK_SIZE = 256;

__global__ void vector_scale(float *a, float scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] *= scalar;
  }
}

int main() {
  // Create an NVTX domain for colored mark events
  nvtxDomainHandle_t domain = nvtxDomainCreateA("NVProfiling");

  // ============================================================
  // Phase 1: Initialize — allocate pinned memory and fill data
  // ============================================================
  {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFF00FF00;  // ARGB green
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Initialize";
    nvtxDomainMarkEx(domain, &attr);
  }
  nvtxRangePushA("Initialize");

  float *h_a;
  CUDA_CHECK(cudaMallocHost(&h_a, VECTOR_SIZE * sizeof(float)));
  for (int i = 0; i < VECTOR_SIZE; i++) {
    h_a[i] = 1.0f;
  }

  float *d_a;
  CUDA_CHECK(cudaMalloc(&d_a, VECTOR_SIZE * sizeof(float)));

  nvtxRangePop();

  // ============================================================
  // Phase 2: Compute — H2D copy, kernel, D2H copy
  // ============================================================
  {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFFFF0000;  // ARGB red
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Compute";
    nvtxDomainMarkEx(domain, &attr);
  }
  nvtxRangePushA("Compute");

  GpuTimer timer;
  timer.start();

  CUDA_CHECK(cudaMemcpy(d_a, h_a, VECTOR_SIZE * sizeof(float),
                        cudaMemcpyHostToDevice));

  int grid = (VECTOR_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
  vector_scale<<<grid, BLOCK_SIZE>>>(d_a, 2.0f, VECTOR_SIZE);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaMemcpy(h_a, d_a, VECTOR_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

  timer.stop();
  float gpu_time = timer.elapsed_ms();

  nvtxRangePop();

  // ============================================================
  // Phase 3: Verify — check results on CPU
  // ============================================================
  {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = 0xFF0000FF;  // ARGB blue
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = "Verify";
    nvtxDomainMarkEx(domain, &attr);
  }
  nvtxRangePushA("Verify");

  bool pass = true;
  int fail_idx = -1;
  for (int i = 0; i < VECTOR_SIZE; i++) {
    if (h_a[i] != 2.0f) {
      pass = false;
      fail_idx = i;
      break;
    }
  }

  nvtxRangePop();

  // ============================================================
  // Report
  // ============================================================
  if (pass) {
    printf("NVTX demo complete. GPU compute time = %.3f ms. "
           "Verification: PASS\n",
           gpu_time);
  } else {
    printf("NVTX demo complete. GPU compute time = %.3f ms. "
           "Verification: FAIL at index %d (got %.1f, expected 2.0)\n",
           gpu_time, fail_idx, h_a[fail_idx]);
  }

  nvtxDomainDestroy(domain);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFreeHost(h_a));

  return 0;
}