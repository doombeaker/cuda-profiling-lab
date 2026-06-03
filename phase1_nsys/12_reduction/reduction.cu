#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../../common/error_check.h"
#include "../../common/timer.h"

const int N = 1 << 24;          // 16M elements
const int BLOCK_SIZE = 256;
const int GRID_SIZE = N / BLOCK_SIZE;  // 65536 blocks

// Interleaved addressing reduction — simple but suffers from warp divergence.
// Threads with large stride become idle in later stages, wasting warp slots.
__global__ void reduce_interleaved(float *input, float *output, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load from global memory into shared memory
  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();

  // Interleaved addressing: stride doubles each step
  // Only threads where tid % (2*stride) == 0 participate
  // This causes warp divergence — half the threads are idle in later stages
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // Thread 0 writes the block's partial sum
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

// Sequential addressing reduction — optimized, avoids warp divergence.
// Threads within a warp access consecutive elements, keeping all threads active.
__global__ void reduce_sequential(float *input, float *output, int n) {
  __shared__ float sdata[BLOCK_SIZE];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load from global memory into shared memory
  sdata[tid] = (i < n) ? input[i] : 0.0f;
  __syncthreads();

  // Sequential addressing: stride halves each step
  // Threads tid < stride participate — consecutive threads, no divergence
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }

  // Thread 0 writes the block's partial sum
  if (tid == 0) {
    output[blockIdx.x] = sdata[0];
  }
}

int main() {
  printf("Reduction (N=%d, sum should be %d.0)\n", N, N);

  // Allocate pinned host memory for input (filled with 1.0f)
  float *h_input;
  CUDA_CHECK(cudaMallocHost(&h_input, N * sizeof(float)));
  for (int i = 0; i < N; i++) {
    h_input[i] = 1.0f;
  }

  // Host buffer for partial results (one float per block)
  float *h_partial;
  CUDA_CHECK(cudaMallocHost(&h_partial, GRID_SIZE * sizeof(float)));

  // Device allocations
  float *d_input, *d_partial;
  CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_partial, GRID_SIZE * sizeof(float)));

  // H2D copy: input data to device
  CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float),
                        cudaMemcpyHostToDevice));

  // --- Kernel 1: Interleaved addressing ---
  GpuTimer timer_interleaved;
  timer_interleaved.start();
  reduce_interleaved<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_partial, N);
  CUDA_CHECK_KERNEL();
  timer_interleaved.stop();
  float interleaved_ms = timer_interleaved.elapsed_ms();

  // D2H partial results
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, GRID_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU-side final reduction on partial sums
  float result_interleaved = 0.0f;
  for (int i = 0; i < GRID_SIZE; i++) {
    result_interleaved += h_partial[i];
  }

  // --- Kernel 2: Sequential addressing ---
  GpuTimer timer_sequential;
  timer_sequential.start();
  reduce_sequential<<<GRID_SIZE, BLOCK_SIZE>>>(d_input, d_partial, N);
  CUDA_CHECK_KERNEL();
  timer_sequential.stop();
  float sequential_ms = timer_sequential.elapsed_ms();

  // D2H partial results
  CUDA_CHECK(cudaMemcpy(h_partial, d_partial, GRID_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // CPU-side final reduction on partial sums
  float result_sequential = 0.0f;
  for (int i = 0; i < GRID_SIZE; i++) {
    result_sequential += h_partial[i];
  }

  // --- Print results ---
  float expected = static_cast<float>(N);
  float error_interleaved = fabsf(result_interleaved - expected);
  float error_sequential = fabsf(result_sequential - expected);
  float speedup = interleaved_ms / sequential_ms;

  printf("reduce_interleaved: GPU = %.3f ms, result = %.1f, error = %.1f\n",
         interleaved_ms, result_interleaved, error_interleaved);
  printf("reduce_sequential:  GPU = %.3f ms, result = %.1f, error = %.1f\n",
         sequential_ms, result_sequential, error_sequential);
  printf("Speedup: sequential is %.2fx faster than interleaved\n", speedup);

  // Verification
  bool pass = (error_interleaved < 1.0f) && (error_sequential < 1.0f);
  if (pass) {
    printf("Verification: PASS (both results within 1.0 of expected)\n");
  } else {
    printf("Verification: FAIL\n");
  }

  // Cleanup
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_partial));
  CUDA_CHECK(cudaFreeHost(h_input));
  CUDA_CHECK(cudaFreeHost(h_partial));

  return pass ? 0 : 1;
}