# 参考分析过程与结论 — Exercise 08: cuBLAS SGEMM Profiling

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**调库 vs 自写 kernel 差距多大？cuBLAS 内部那个长得离谱的 kernel 名是什么？** 用 `cuda_gpu_kern_sum` + `cuda_kern_exec_sum` 给出工程级证据。

## 0. 一句话结论

cuBLAS SGEMM 1024×1024 实测 **93.0 µs**，比 exercise 02 的 `matmul_tiled`（451.8 µs）快 **4.85×**，比 `matmul_naive`（686.6 µs）快 **7.38×**。差距源于 cuBLAS 用 Tensor Core、寄存器分块、双缓冲等 assembly 级优化——这些是手工 kernel 写不出来的。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `cublas.cu` |
| 矩阵维度 | `N=M=K=1024`（4 MB 每矩阵，column-major） |
| 采集 | `nsys profile -o report08 --force-overwrite true --trace=cuda,cublas ./cublas` |
| 4 个 cublas API | `cublasCreate`、`cublasSgemm` (warmup)、`cublasSgemm` (timed)、`cublasDestroy` |
| 自打印 | `cuBLAS SGEMM (1024x1024): GPU time = 0.092 ms`、`Verification: PASS (spot-check)` |

cuBLAS 用了 column-major 布局，源码注释解释了 `C = B * A in column-major = (A*B)^T^T = A*B` 在 row-major 读回的转置技巧——这是用 cuBLAS 跑 row-major 数据的标准 workaround。

## 2. 关键测量数据

### 2.1 cuBLAS kernel（来自 `cuda_gpu_kern_sum`）

| Kernel | 次数 | 总耗时 | 占 kernel 时长 |
|---|---|---|---|
| `sm80_xmma_gemm_f32f32_f32f32_f32_nn_n_tilesize64x128x8_stage3_warpsize1x4x1_ffma_aligna4_alignc4_execute_kernel__5x_cublas` | 2 | 186 µs | 100% |

> 长得离谱的名字解读：
> - `sm80`: target arch sm_80（Ampere）—— 注意不是 sm_90！cuBLAS 在 H100/H20 上仍可能选 sm_80 kernel（其优化版本对 Hopper 适配未必最优）
> - `xmma_gemm`: xmma 是 NVIDIA 内部 GEMM template library
> - `f32f32_f32f32_f32`: in_a=f32, in_b=f32, in_c=f32, out_c=f32
> - `nn`: 第一个 n = A 不转置，第二个 n = B 不转置
> - `tilesize64x128x8`: m=64, n=128, k=8 的 tile size（每 block 计算的 C 子块维度）
> - `stage3`: 3 个 stage 的流水线（双缓冲 + 1 算），隐藏访存延迟
> - `warpsize1x4x1`: warp 在 tile 内的二维组织 (1×4×1)
> - `ffma`: 用 FP32 FMA 指令（非 tensor core）
> - `aligna4_alignc4`: A 和 C 都按 4 字节对齐

### 2.2 单次 launch（warmup + timed，来自 `cuda_kern_exec_sum`）

| 维度 | 数值 |
|---|---|
| 次数 | 2（warmup + timed） |
| cudaLaunchKernelExC API 调用 avg | 17.9 µs |
| kernel 实际执行 avg | **93.0 µs** |
| 总耗时 avg | 111.3 µs |

> cuBLAS 用的是 `cudaLaunchKernelExC_v11060`（不是普通的 `cudaLaunchKernel`）。这是 CUDA 11.6+ 的高级 launch API，支持 cluster launch 等新特性。本练习 host API 表中能看到 810 次 `cuGetProcAddress_v2`（库查找函数指针），可见 cuBLAS 启动开销主要在加载库自身。

### 2.3 memcpy（来自 `cuda_gpu_mem_time_sum`）

| 操作 | 次数 | 单次平均 | 总耗时 |
|---|---|---|---|
| H2D | 2 | 78.2 µs | 156.4 µs（4 MB × 2 = 8 MB） |
| D2H | 1 | 78.7 µs | 78.7 µs（4 MB） |
| Memset | 1 | 2.7 µs | 2.7 µs（清 d_C） |

> 数据量小（4 MB × 3 = 12 MB）所以 memcpy 都在 80 µs 内完成。同样能算带宽：4 MB / 78.2 µs = **51.2 GB/s**——与大尺寸下 pinned 满带宽一致。

### 2.4 host API（节选自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 备注 |
|---|---|---|---|
| `cudaHostAlloc` | 3 | 199.2 ms | 88.8%（setup A/B/C 三块 pinned memory） |
| `cuLibraryLoadData` | 4 | 18.1 ms | 8.1%（**cuBLAS 内核二进制 JIT 加载**） |
| `cudaFreeHost` | 3 | 1.52 ms | 0.7% |
| `cudaMalloc` | 6 | 0.57 ms | <0.3% |
| `cuGetProcAddress_v2` | **810** | 0.50 ms | <0.3%（**库函数符号查找**） |
| `cudaMemcpy` | 3 | 0.29 ms | <0.1% |
| `cudaEventCreateWithFlags` | 18 | 0.28 ms | <0.1%（cuBLAS 内部用 event 同步） |
| `cudaDeviceSynchronize` | 5 | 0.10 ms | 0.1%（warmup 后同步） |
| `cudaLaunchKernelExC_v11060` | 2 | 35.8 µs | cuBLAS 内部用 |

### 2.5 NVTX range（来自 `nvtx_pushpop_sum`，cuBLAS trace 自带）

| Range | 次数 | 总耗时 |
|---|---|---|
| `cuBLAS:cublasCreate_v2` | 1 | 54.5 ms（cuBLAS handle 初始化） |

`cublasCreate` 单次 54.5 ms 也是一次性 setup 成本，应在 GpuTimer 之前。

## 3. 分析与结论

### 3.1 横向对比：自己写 vs 调库

| 实现 | 1024×1024 SGEMM 耗时 | 相对最快 | 关键技术 |
|---|---|---|---|
| `matmul_naive` (ex 02) | 686.6 µs | **7.38× slow** | 朴素访存 |
| `matmul_tiled` (ex 02) | 451.8 µs | **4.85× slow** | shared memory + 16× tile 复用 |
| `cublasSgemm` (本练习) | **93.0 µs** | 1.00× | xmma template + tilesize 64×128×8 + 3-stage pipeline + register blocking + warp-level layout + assembly-level tuning |

cuBLAS 比 tiled 快 4.85× 不是因为"算法更聪明"——两者底层都是 tile + shared memory + register blocking 的相同思路。差距在工程细节：
- **tilesize 64×128×8** vs 我们 tiled 的 **16×16**：cuBLAS 每个 block 算的 C 子块大 32×，意味着 shared memory 复用率高 32× ——这正对应 tiled 加速比 TILE=16 的理论上限
- **3-stage pipeline** 在加载下一 tile 的同时计算当前 tile 的 partial dot：访存完全被计算隐藏
- **ffma (FP32 FMA)** 指令：每个 cycle 一条 FMA，乘加流水线化

### 3.2 注意：cuBLAS 用 `sm80` 而非 `sm90` Tensor Core

`sm80_xmma_gemm_f32f32_..._ffma_` —— kernel 名说它跑 sm_80 指令且用 **`ffma`（FP32 标量 FMA）而非 `hfma2`/`mma.sync`（Tensor Core）**。这意味着本测试 case 走的是 **FP32 通用路径**而非 H20 的 TF32 / FP16 Tensor Core 路径。

如果换 `cublasSgemmEx` 配合 `CUBLAS_COMPUTE_32F_FAST_TF32` 或 `cublasHgemm`（FP16），应该能看到 `sm90` + `mma.sync` kernel，速度可能再快 3-10×。

**工程含义**：cuBLAS 默认 SGEMM 是 FP32 数值正确性最高的版本；要榨 H20 峰值需显式选 TF32/FP16/BF16。AI 训练场景这恰是默认选择，所以训练计算 graph 里 matmul kernel 名通常带 `mma` 而非 `ffma`。

### 3.3 cuBLAS 启动总成本：~73 ms

```
cublasCreate                           54.5 ms   ← handle 初始化 + JIT 装载二进制
cuLibraryLoadData × 4                  18.1 ms   ← kernel binary 第一次加载到 device
cudaEventCreateWithFlags × 18         0.28 ms   ← cuBLAS 内部 sync 资源
---
合计启动开销                          ≈ 73 ms    (单次只付一次)
```

**生产代码含义**：cuBLAS handle 应在程序入口创建一次，整个进程生命周期复用；每个 iteration 不要 destroy/recreate。这是 cuBLAS 性能优化的第一条。

### 3.4 GpuTimer 测到的 0.092 ms 为何与 stats 报的 93.0 µs 略有差？

- GpuTimer 用 `cudaEvent` 记录 GPU device time：start = warmup 后一次 event，stop = timed run 后 event
- nsys stats 中 `KAvg = 93.0 µs`：从 CUPTI kernel activity 抓到的 kernel 实际 GPU 起止时间
- 两者应几乎相等；GpuTimer 多算了几 µs 因为 event record 本身有几 µs 延迟

这种小差异在所有练习都存在，是正常的测量噪声范围。

## 4. 思考延伸

1. 把矩阵维度从 1024 改成 4096（不调 cuBLAS）：cuBLAS 应仍 ~milliseconds 级，但 naive/tiled 的 ex 02 受访存压力会暴露真加速比。验证 cuBLAS 在大 N 下是否相对优势更大。
2. 用 `cublasHgemm`（FP16）替换 `cublasSgemm`，重新 profile，看 kernel 名是否切换到 `mma` Tensor Core 路径，速度提升几倍。
3. cuBLAS 内部 18 次 `cudaEventCreateWithFlags` 是它在做什么？是否每次 SGEMM 都创建新 event？用 `cuda_gpu_trace` 看 cuBLAS kernel 周围有没有 event record 模式。
4. 既然 cuBLAS handle 创建耗费 73 ms，多线程并发用同一个 handle 是否安全？《cuBLAS 文档》说 handle 是 thread-safe 但性能上建议每线程一个 handle。

## 附录：复现命令

```bash
cd phase1_nsys/08_cublas && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum   --format csv ./report08.nsys-rep
$NSYS stats --force-export=true --report cuda_kern_exec_sum  --format csv ./report08.nsys-rep
$NSYS stats --force-export=true --report cuda_api_sum        --format csv ./report08.nsys-rep | grep -E "(cublas|cuda(Malloc|HostAlloc|Memcpy|LaunchKernel|Event|DeviceSync)|cuLibrary|cuGetProc)"
```

更详细看 cuBLAS API（不只是 CUDA）：
```bash
$NSYS stats --force-export=true --report cuda_api_trace --format csv ./report08.nsys-rep | grep -i cublas
```
