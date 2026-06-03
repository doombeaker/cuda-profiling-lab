#include "../../common/error_check.h"
#include "../../common/timer.h"

constexpr int N = 1 << 16;          // 64K elements — 256 blocks, plenty of room for concurrent kernels
constexpr int ITERATIONS = 200000;  // enough iterations for ~1-10ms kernel time
constexpr int BLOCK_SIZE = 256;
constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

// Spins for `iterations` loops doing a trivial computation in registers to
// create measurable GPU work time. Each thread processes one element.
// Uses register-only computation to be compute-bound, allowing two kernels
// to overlap on different SMs.
__global__ void delay_kernel(float *data, int n, int iterations) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float val = data[idx];
  for (int i = 0; i < iterations; i++) {
    val = val * 0.999f + 0.001f;
  }
  data[idx] = val;
}

int main() {
  // Allocate pinned host memory and initialize
  float *h_data;
  CUDA_CHECK(cudaMallocHost(&h_data, N * sizeof(float)));
  for (int i = 0; i < N; i++) {
    h_data[i] = 1.0f;
  }

  // Allocate two device arrays — one per stream — to avoid data races when
  // streams run concurrently
  float *d_data1, *d_data2;
  CUDA_CHECK(cudaMalloc(&d_data1, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_data2, N * sizeof(float)));

  // Copy initial data to both device arrays
  CUDA_CHECK(cudaMemcpy(d_data1, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_data2, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

  // Create two streams
  cudaStream_t stream1, stream2;
  CUDA_CHECK(cudaStreamCreate(&stream1));
  CUDA_CHECK(cudaStreamCreate(&stream2));

  CpuTimer cpu_timer;
  double t_device, t_stream, t_event;

  // =========================================================================
  // Test 1: cudaDeviceSynchronize() — blocks ALL streams
  // =========================================================================
  printf("=== Test 1: cudaDeviceSynchronize() ===\n");
  printf("Launch kernel on stream1, then DeviceSynchronize (blocks ALL streams),\n");
  printf("then launch kernel on stream2. Kernels are serialized.\n\n");

  cpu_timer.start();
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_data1, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaDeviceSynchronize());  // blocks CPU until ALL GPU work is done
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream2>>>(d_data2, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaDeviceSynchronize());  // wait for stream2 to finish
  cpu_timer.stop();
  t_device = cpu_timer.elapsed_ms();
  printf("DeviceSynchronize: total time = %.3f ms\n\n", t_device);

  // =========================================================================
  // Test 2: cudaStreamSynchronize(stream) — blocks only one stream
  // =========================================================================
  printf("=== Test 2: cudaStreamSynchronize() ===\n");
  printf("Launch kernels on stream1 and stream2 concurrently (no sync between),\n");
  printf("then StreamSynchronize each. Kernels can overlap on GPU.\n\n");

  cpu_timer.start();
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_data1, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream2>>>(d_data2, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaStreamSynchronize(stream1));  // blocks only stream1
  CUDA_CHECK(cudaStreamSynchronize(stream2));  // blocks only stream2
  cpu_timer.stop();
  t_stream = cpu_timer.elapsed_ms();
  printf("StreamSynchronize: total time = %.3f ms\n\n", t_stream);

  // =========================================================================
  // Test 3: cudaEventSynchronize() — finest granularity
  // =========================================================================
  printf("=== Test 3: cudaEventSynchronize() ===\n");
  printf("Launch kernels on stream1 and stream2 concurrently, record events,\n");
  printf("then EventSynchronize each event. Finest granularity — wait for a\n");
  printf("specific point in a specific stream.\n\n");

  cudaEvent_t e1, e2;
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventCreate(&e2));

  cpu_timer.start();
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream1>>>(d_data1, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaEventRecord(e1, stream1));  // record event after stream1's kernel
  delay_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, stream2>>>(d_data2, N, ITERATIONS);
  CUDA_CHECK_KERNEL();
  CUDA_CHECK(cudaEventRecord(e2, stream2));  // record event after stream2's kernel
  CUDA_CHECK(cudaEventSynchronize(e1));      // wait for stream1's kernel only
  CUDA_CHECK(cudaEventSynchronize(e2));      // wait for stream2's kernel only
  cpu_timer.stop();
  t_event = cpu_timer.elapsed_ms();
  printf("EventSynchronize: total time = %.3f ms\n\n", t_event);

  // =========================================================================
  // Summary
  // =========================================================================
  printf("=== Summary ===\n");
  printf("%-25s %s\n", "Method", "Total Time (ms)");
  printf("-----------------------------------------\n");
  printf("%-25s %.3f\n", "DeviceSynchronize", t_device);
  printf("%-25s %.3f\n", "StreamSynchronize", t_stream);
  printf("%-25s %.3f\n", "EventSynchronize", t_event);
  printf("\n");
  printf("Expected: DeviceSynchronize ≈ 2× kernel time (serialized).\n");
  printf("          StreamSynchronize ≈ 1× kernel time (overlapped).\n");
  printf("          EventSynchronize  ≈ 1× kernel time (overlapped).\n");

  // =========================================================================
  // Cleanup
  // =========================================================================
  CUDA_CHECK(cudaEventDestroy(e1));
  CUDA_CHECK(cudaEventDestroy(e2));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaStreamDestroy(stream2));
  CUDA_CHECK(cudaFree(d_data1));
  CUDA_CHECK(cudaFree(d_data2));
  CUDA_CHECK(cudaFreeHost(h_data));

  return 0;
}