#pragma once
#include <chrono>
#include <cstdio>
#include "error_check.h"

class GpuTimer {
public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~GpuTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(start_, stream));
  }

  void stop(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaEventRecord(stop_, stream));
  }

  float elapsed_ms() const {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop_));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }

private:
  cudaEvent_t start_, stop_;
};

class CpuTimer {
public:
  void start() { t0_ = std::chrono::high_resolution_clock::now(); }
  void stop() { t1_ = std::chrono::high_resolution_clock::now(); }
  double elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(t1_ - t0_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point t0_, t1_;
};