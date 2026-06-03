# Exercise 08: cuBLAS Profiling

## Learning Goal
Profile a real GPU library (cuBLAS) with nsys. See how library kernels appear on the timeline and compare library performance with hand-written code.

## Prerequisite
Exercise 02 (matmul comparison), basic nsys profiling

## Key Concepts
- cuBLAS is NVIDIA's optimized BLAS library — SGEMM is highly tuned for each GPU architecture
- Library calls appear as cudaLaunchKernel API calls on the CPU timeline, just like hand-written kernels
- Library kernels often have cryptic names (e.g., "ampere_sgemm_128x128_nn") — identifiable by the cublas namespace on the timeline
- Comparing cuBLAS SGEMM vs exercise 02 tiled matmul: library is typically 2-5x faster due to assembly-level tuning
- The nsys timeline also shows cublasCreate/cublasDestroy as CUDA API calls

## Build & Run
```
make && ./profile.sh
```

## nsys Flags Explained
`--trace=cuda,cublas` captures both CUDA API and cuBLAS API calls on the timeline

## How to Read Results

### Command-line Output
Compare cuBLAS timing with Exercise 02 tiled matmul timing (run separately).

### nsys GUI
1. Open `nsys-ui ./report08.nsys-rep`
2. Look at the CUDA API row: find `cublasSgemm_v2` API call — this is the host-side dispatch
3. On the GPU timeline, find the cuBLAS kernel — it may appear with a name like "ampere_sgemm_*" in the kernel row
4. The cuBLAS kernel will likely be a single thick green bar (unlike tiled matmul which launches many blocks)
5. Check the CUDA Memory row for H2D/D2H operations

## Experiment
Run Exercise 02 matmul_tiled again and compare the nsys timeline side by side with this cuBLAS version.