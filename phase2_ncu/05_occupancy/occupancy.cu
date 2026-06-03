#include "../../common/error_check.h"
#include "../../common/timer.h"
#include <cmath>

__global__ void kernel_low_reg(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void __launch_bounds__(256) kernel_medium_reg(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v0 = a[i]; float v1 = b[i];
        float v2 = a[i] * 0.1f; float v3 = b[i] * 0.1f;
        float v4 = a[i] * 0.2f; float v5 = b[i] * 0.2f;
        float v6 = a[i] * 0.3f; float v7 = b[i] * 0.3f;
        float v8 = a[i] * 0.4f; float v9 = b[i] * 0.4f;
        float v10 = a[i] * 0.5f; float v11 = b[i] * 0.5f;
        float v12 = a[i] * 0.6f; float v13 = b[i] * 0.6f;
        float v14 = a[i] * 0.7f; float v15 = b[i] * 0.7f;
        float v16 = a[i] * 0.8f; float v17 = b[i] * 0.8f;
        float v18 = a[i] * 0.9f; float v19 = b[i] * 0.9f;
        float v20 = a[i] * 1.1f; float v21 = b[i] * 1.1f;
        float v22 = a[i] * 1.2f; float v23 = b[i] * 1.2f;
        float v24 = a[i] * 1.3f; float v25 = b[i] * 1.3f;
        float v26 = a[i] * 1.4f; float v27 = b[i] * 1.4f;
        float v28 = a[i] * 1.5f; float v29 = b[i] * 1.5f;
        float v30 = a[i] * 1.6f; float v31 = b[i] * 1.6f;
        float v32 = a[i] * 1.7f; float v33 = b[i] * 1.7f;
        float v34 = a[i] * 1.8f; float v35 = b[i] * 1.8f;
        float v36 = a[i] * 1.9f; float v37 = b[i] * 1.9f;
        float v38 = a[i] * 2.0f; float v39 = b[i] * 2.0f;
        c[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
             + v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19
             + v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28 + v29
             + v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37 + v38 + v39;
    }
}

__global__ void __launch_bounds__(256) kernel_high_reg(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v0 = a[i]; float v1 = b[i];
        float v2 = a[i] * 0.01f; float v3 = b[i] * 0.01f;
        float v4 = a[i] * 0.02f; float v5 = b[i] * 0.02f;
        float v6 = a[i] * 0.03f; float v7 = b[i] * 0.03f;
        float v8 = a[i] * 0.04f; float v9 = b[i] * 0.04f;
        float v10 = a[i] * 0.05f; float v11 = b[i] * 0.05f;
        float v12 = a[i] * 0.06f; float v13 = b[i] * 0.06f;
        float v14 = a[i] * 0.07f; float v15 = b[i] * 0.07f;
        float v16 = a[i] * 0.08f; float v17 = b[i] * 0.08f;
        float v18 = a[i] * 0.09f; float v19 = b[i] * 0.09f;
        float v20 = a[i] * 0.10f; float v21 = b[i] * 0.10f;
        float v22 = a[i] * 0.11f; float v23 = b[i] * 0.11f;
        float v24 = a[i] * 0.12f; float v25 = b[i] * 0.12f;
        float v26 = a[i] * 0.13f; float v27 = b[i] * 0.13f;
        float v28 = a[i] * 0.14f; float v29 = b[i] * 0.14f;
        float v30 = a[i] * 0.15f; float v31 = b[i] * 0.15f;
        float v32 = a[i] * 0.16f; float v33 = b[i] * 0.16f;
        float v34 = a[i] * 0.17f; float v35 = b[i] * 0.17f;
        float v36 = a[i] * 0.18f; float v37 = b[i] * 0.18f;
        float v38 = a[i] * 0.19f; float v39 = b[i] * 0.19f;
        float v40 = a[i] * 0.20f; float v41 = b[i] * 0.20f;
        float v42 = a[i] * 0.21f; float v43 = b[i] * 0.21f;
        float v44 = a[i] * 0.22f; float v45 = b[i] * 0.22f;
        float v46 = a[i] * 0.23f; float v47 = b[i] * 0.23f;
        float v48 = a[i] * 0.24f; float v49 = b[i] * 0.24f;
        float v50 = a[i] * 0.25f; float v51 = b[i] * 0.25f;
        float v52 = a[i] * 0.26f; float v53 = b[i] * 0.26f;
        float v54 = a[i] * 0.27f; float v55 = b[i] * 0.27f;
        float v56 = a[i] * 0.28f; float v57 = b[i] * 0.28f;
        float v58 = a[i] * 0.29f; float v59 = b[i] * 0.29f;
        float v60 = a[i] * 0.30f; float v61 = b[i] * 0.30f;
        float v62 = a[i] * 0.31f; float v63 = b[i] * 0.31f;
        float v64 = a[i] * 0.32f; float v65 = b[i] * 0.32f;
        float v66 = a[i] * 0.33f; float v67 = b[i] * 0.33f;
        float v68 = a[i] * 0.34f; float v69 = b[i] * 0.34f;
        float v70 = a[i] * 0.35f; float v71 = b[i] * 0.35f;
        float v72 = a[i] * 0.36f; float v73 = b[i] * 0.36f;
        float v74 = a[i] * 0.37f; float v75 = b[i] * 0.37f;
        float v76 = a[i] * 0.38f; float v77 = b[i] * 0.38f;
        float v78 = a[i] * 0.39f; float v79 = b[i] * 0.39f;
        c[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9
             + v10 + v11 + v12 + v13 + v14 + v15 + v16 + v17 + v18 + v19
             + v20 + v21 + v22 + v23 + v24 + v25 + v26 + v27 + v28 + v29
             + v30 + v31 + v32 + v33 + v34 + v35 + v36 + v37 + v38 + v39
             + v40 + v41 + v42 + v43 + v44 + v45 + v46 + v47 + v48 + v49
             + v50 + v51 + v52 + v53 + v54 + v55 + v56 + v57 + v58 + v59
             + v60 + v61 + v62 + v63 + v64 + v65 + v66 + v67 + v68 + v69
             + v70 + v71 + v72 + v73 + v74 + v75 + v76 + v77 + v78 + v79;
    }
}

int main() {
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    float *d_a, *d_b, *d_c_low, *d_c_med, *d_c_high;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_low, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_med, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_high, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blocks = (N + 255) / 256;
    int threads = 256;

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer0;
    timer0.start();
    kernel_low_reg<<<blocks, threads>>>(d_a, d_b, d_c_low, N);
    CUDA_CHECK_KERNEL();
    timer0.stop();
    float ms_low = timer0.elapsed_ms();

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer1;
    timer1.start();
    kernel_medium_reg<<<blocks, threads>>>(d_a, d_b, d_c_med, N);
    CUDA_CHECK_KERNEL();
    timer1.stop();
    float ms_medium = timer1.elapsed_ms();

    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer timer2;
    timer2.start();
    kernel_high_reg<<<blocks, threads>>>(d_a, d_b, d_c_high, N);
    CUDA_CHECK_KERNEL();
    timer2.stop();
    float ms_high = timer2.elapsed_ms();

    std::printf("low_reg:    %.3f ms\n", ms_low);
    std::printf("medium_reg: %.3f ms\n", ms_medium);
    std::printf("high_reg:   %.3f ms\n", ms_high);
    std::printf("Expected occupancy: low_reg > medium_reg > high_reg (check with ncu)\n");

    float expected_low = 1.0f + 2.0f;
    CUDA_CHECK(cudaMemcpy(h_c, d_c_low, bytes, cudaMemcpyDeviceToHost));
    bool pass_low = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - expected_low) > 1e-3f) { pass_low = false; break; }
    }
    std::printf("low_reg verification: %s\n", pass_low ? "PASS" : "FAIL");

    float expected_medium = 1.0f + 2.0f;
    for (int k = 1; k <= 9; k++) expected_medium += (1.0f + 2.0f) * (0.1f * k);
    for (int k = 11; k <= 20; k++) expected_medium += (1.0f + 2.0f) * (0.1f * k);
    CUDA_CHECK(cudaMemcpy(h_c, d_c_med, bytes, cudaMemcpyDeviceToHost));
    bool pass_medium = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - expected_medium) > 0.5f) { pass_medium = false; break; }
    }
    std::printf("medium_reg verification: %s\n", pass_medium ? "PASS" : "FAIL");

    float expected_high = 1.0f + 2.0f;
    for (int k = 1; k < 40; k++) expected_high += (1.0f + 2.0f) * (0.01f * k);
    CUDA_CHECK(cudaMemcpy(h_c, d_c_high, bytes, cudaMemcpyDeviceToHost));
    bool pass_high = true;
    for (int i = 0; i < N; i++) {
        if (std::fabs(h_c[i] - expected_high) > 0.5f) { pass_high = false; break; }
    }
    std::printf("high_reg verification: %s\n", pass_high ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c_low));
    CUDA_CHECK(cudaFree(d_c_med));
    CUDA_CHECK(cudaFree(d_c_high));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
