#pragma once
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(err)                                                        \
  do {                                                                         \
    cudaError_t err_ = (err);                                                  \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__,        \
                   __LINE__, cudaGetErrorString(err_), err_);                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t err_ = cudaGetLastError();                                     \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "Kernel launch error at %s:%d: %s (%d)\n",         \
                   __FILE__, __LINE__, cudaGetErrorString(err_), err_);        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
