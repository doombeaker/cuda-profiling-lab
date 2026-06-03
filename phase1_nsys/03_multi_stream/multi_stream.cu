#include "../../common/error_check.h"
#include "../../common/timer.h"

constexpr int NUM_STREAMS = 4;
constexpr int CHUNK_SIZE = 1 << 24;
constexpr int TOTAL_SIZE = NUM_STREAMS * CHUNK_SIZE;
constexpr int BLOCK_SIZE = 256;
constexpr float VECTOR_SCALE = 2.0f;

__global__ void vector_scale(float *data, float scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] *= scalar;
  }
}

int main() {
  float *h_data;
  CUDA_CHECK(cudaMallocHost(&h_data, TOTAL_SIZE * sizeof(float)));

  for (int i = 0; i < TOTAL_SIZE; i++) {
    h_data[i] = (float)i;
  }

  float *d_data;
  CUDA_CHECK(cudaMalloc(&d_data, TOTAL_SIZE * sizeof(float)));

  cudaStream_t streams[NUM_STREAMS];
  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  cudaEvent_t start_event, stop_event;
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  CUDA_CHECK(cudaEventRecord(start_event));

  for (int i = 0; i < NUM_STREAMS; i++) {
    int offset = i * CHUNK_SIZE;
    CUDA_CHECK(cudaMemcpyAsync(d_data + offset, h_data + offset,
                               CHUNK_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice, streams[i]));
    int grid = (CHUNK_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_scale<<<grid, BLOCK_SIZE, 0, streams[i]>>>(d_data + offset,
                                                       VECTOR_SCALE,
                                                       CHUNK_SIZE);
    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaMemcpyAsync(h_data + offset, d_data + offset,
                               CHUNK_SIZE * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[i]));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(stop_event));
  CUDA_CHECK(cudaEventSynchronize(stop_event));
  float elapsed = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_event, stop_event));
  printf("Multi-Stream (4 streams): total time = %.3f ms\n", elapsed);

  bool pass = true;
  int fail_idx = -1;
  for (int i = 0; i < TOTAL_SIZE; i++) {
    float expected = (float)i * VECTOR_SCALE;
    if (h_data[i] != expected) {
      pass = false;
      fail_idx = i;
      break;
    }
  }

  if (pass) {
    printf("Verification: PASS\n");
  } else {
    printf("Verification: FAIL at index %d (got %.1f, expected %.1f)\n",
           fail_idx, h_data[fail_idx], (float)fail_idx * VECTOR_SCALE);
  }

  for (int i = 0; i < NUM_STREAMS; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaFree(d_data));
  CUDA_CHECK(cudaFreeHost(h_data));

  return 0;
}
