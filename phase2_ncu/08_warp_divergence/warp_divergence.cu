#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// Interleaved addressing reduction — causes severe warp divergence.
// Threads with tid % (2*s) == 0 participate; as stride grows,
// fewer threads per warp are active, serializing execution.
__global__ void divergent_reduce(float *input, float *output, int n) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Sequential addressing reduction — no warp divergence.
// Threads tid < s participate; consecutive threads within a warp
// take the same branch, keeping all lanes active.
__global__ void uniform_reduce(float *input, float *output, int n) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Branch-on-threshold kernel — splits work at a threshold.
// Half the elements are above threshold, half below,
// producing ~50% branch efficiency.
__global__ void branch_threshold(float *data, int n, float threshold) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (data[i] > threshold) {
    data[i] = sqrtf(data[i]);
  } else {
    data[i] = data[i] * data[i];
  }
}

int main() {
  const int N = 1 << 24;           // 16M elements
  const int BLOCK_SIZE = 256;
  const int GRID_SIZE = N / BLOCK_SIZE;  // 65536 blocks
  const float THRESHOLD = 0.5f;

  size_t bytes = N * sizeof(float);
  size_t partial_bytes = GRID_SIZE * sizeof(float);

  // Pinned host memory for input
  float *h_input;
  CUDA_CHECK(cudaMallocHost(&h_input, bytes));

  // Fill with alternating above/below threshold values
  for (int i = 0; i < N; i++) {
    h_input[i] = (i % 2 == 0) ? 0.0f : 1.0f;  // below : above threshold
  }

  // Pinned host memory for partial reduction results
  float *h_partial;
  CUDA_CHECK(cudaMallocHost(&h_partial, partial_bytes));

  // Device allocations
  float *d_input, *d_partial, *d_data;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_partial, partial_bytes));
  CUDA_CHECK(cudaMalloc(&d_data, bytes));

  // --- Kernel 1: divergent_reduce ---
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer_div;
  timer_div.start();
  divergent_reduce<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_partial, N);
  CUDA_CHECK_KERNEL();
  timer_div.stop();
  float ms_div = timer_div.elapsed_ms();

  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, partial_bytes,
                        cudaMemcpyDeviceToHost));
  float sum_div = 0.0f;
  for (int i = 0; i < GRID_SIZE; i++) sum_div += h_partial[i];

  // --- Kernel 2: uniform_reduce ---
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer_uni;
  timer_uni.start();
  uniform_reduce<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_partial, N);
  CUDA_CHECK_KERNEL();
  timer_uni.stop();
  float ms_uni = timer_uni.elapsed_ms();

  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, partial_bytes,
                        cudaMemcpyDeviceToHost));
  float sum_uni = 0.0f;
  for (int i = 0; i < GRID_SIZE; i++) sum_uni += h_partial[i];

  // --- Kernel 3: branch_threshold ---
  CUDA_CHECK(cudaMemcpy(d_data, h_input, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  GpuTimer timer_br;
  timer_br.start();
  branch_threshold<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, N, THRESHOLD);
  CUDA_CHECK_KERNEL();
  timer_br.stop();
  float ms_br = timer_br.elapsed_ms();

  // Copy branch_threshold result back for verification
  float *h_result;
  CUDA_CHECK(cudaMallocHost(&h_result, bytes));
  CUDA_CHECK(cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost));

  // --- Print timing ---
  std::printf("divergent_reduce: %.3f ms\n", ms_div);
  std::printf("uniform_reduce: %.3f ms\n", ms_uni);
  std::printf("branch_threshold: %.3f ms\n", ms_br);

  // --- Verify reduction results ---
  float expected_sum = static_cast<float>(N / 2);  // half are 1.0f
  float err_div = std::fabs(sum_div - expected_sum);
  float err_uni = std::fabs(sum_uni - expected_sum);
  bool pass_div = (err_div < 1.0f);
  bool pass_uni = (err_uni < 1.0f);

  std::printf("divergent_reduce sum: %.1f (expected %.1f, error %.1f) %s\n",
              sum_div, expected_sum, err_div, pass_div ? "PASS" : "FAIL");
  std::printf("uniform_reduce sum:   %.1f (expected %.1f, error %.1f) %s\n",
              sum_uni, expected_sum, err_uni, pass_uni ? "PASS" : "FAIL");

  // --- Verify branch_threshold results ---
  // Above threshold (1.0f): sqrtf(1.0f) = 1.0f
  // Below threshold (0.0f): 0.0f * 0.0f = 0.0f
  bool pass_br = true;
  for (int i = 0; i < N; i++) {
    float expected = (i % 2 == 0) ? 0.0f : 1.0f;
    if (std::fabs(h_result[i] - expected) > 1e-4f) {
      pass_br = false;
      break;
    }
  }
  std::printf("branch_threshold verification: %s\n", pass_br ? "PASS" : "FAIL");

  // --- Cleanup ---
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFreeHost(h_input));
  CUDA_CHECK(cudaFreeHost(h_partial));
  CUDA_CHECK(cudaFreeHost(h_result));

  bool all_pass = pass_div && pass_uni && pass_br;
  return all_pass ? 0 : 1;
}