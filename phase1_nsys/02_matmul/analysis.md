# 参考分析过程与结论 — Exercise 02: Matmul Naive vs Tiled

> 沿用 exercise 01 的 AI 友好路径（SQLite + CSV）。本文聚焦"naive vs tiled 倍数对比"这一个具体命题，验证 README 宣称的"tiled 快好几倍"是否成立。

## 0. 一句话结论

实测 **tiled 仅比 naive 快 1.52×**（451µs vs 686µs），**远低于** README 宣称与教学预期的"several times faster"。原因是 `N=1024` 太小，naive 也能借 H20 巨大 L2 cache 抵消大部分全局访存低效。要看真正 tiled 优势，需把 N 提升至 ≥ 4096。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `matmul.cu` |
| 矩阵规模 | `N = 1024`（每矩阵 4 MB） |
| Tile 大小 | `TILE = 16`（每个 block 16×16 线程） |
| Grid 配置 | `dim3(N/16, N/16) = 64×64` blocks |
| Host 内存 | pinned (`cudaMallocHost`) |
| 验证 | spot-check 5 个随机元素，通过 |
| 采集 | `nsys profile -o report02 --force-overwrite true ./matmul` |

## 2. 关键测量数据

### 2.1 两 kernel 耗时（来自 `cuda_gpu_kern_sum`）

| Kernel | 次数 | 总耗时 | 占 kernel 总时长 |
|---|---|---|---|
| `matmul_naive` | 1 | **686.6 µs** | 60.3% |
| `matmul_tiled` | 1 | **451.8 µs** | 39.7% |
| **加速比** | — | `686.6 / 451.8 = 1.52×` | — |

### 2.2 memcpy 时长（来自 `cuda_gpu_mem_time_sum`）

| 操作 | 次数 | 单次 | 总耗时 |
|---|---|---|---|
| H2D | 2 | 86.4 µs | 172.9 µs（4 MB × 2） |
| D2H | 1 | 77.9 µs | 77.9 µs（4 MB × 1） |
| memset | 1 | 2.5 µs | 2.5 µs（清零 d_C 之间） |

### 2.3 host API 时长（来自 `cuda_api_sum`，节选）

| API | 次数 | 总耗时 |
|---|---|---|
| `cudaHostAlloc` | 3 | 198.8 ms（setup，**98.2% host 总时长**） |
| `cudaEventSynchronize` | 2 | 1.15 ms（GpuTimer 内部） |
| `cudaMemcpy` | 3 | 0.30 ms |
| `cudaLaunchKernel` | 2 | 0.14 ms |
| `cudaMemset` | 1 | 10 µs |

### 2.4 launch vs kernel 时长（来自 `cuda_kern_exec_sum`）

| Kernel | cudaLaunchKernel 调用 | queue 等待 | kernel 实际 |
|---|---|---|---|
| `matmul_naive` | 122.6 µs | — | 686.6 µs |
| `matmul_tiled` | 15.3 µs | 2.0 µs | 451.8 µs |

> tiled 的 `cudaLaunchKernel` 仅 15.3 µs（明显比 naive 的 122.6 µs 快）——这是 host 端驱动对不同 kernel 配置/参数的派发开销差异。

## 3. 分析与结论

### 3.1 1.52× 的实测与 16× 的理论：差距从哪里来？

**理论加速比 = TILE = 16×**（每个 tile element 被复用 TILE 次，全局访存减少 16×）。实测 1.52×，**仅发挥了 9.5%**。

| 失分原因 | 解释 | 占比估计 |
|---|---|---|
| **L2 cache 黑魔法** | H20 L2 高达数十 MB；3 个 4 MB 矩阵共 12 MB 完全驻留在 L2。naive kernel 每次访问 A/B 实际走 L2（~80 cycle）而非 HBM（~400 cycle），原本"全局访存贵 100×"的假设被推翻 | 主要失分 |
| **__syncthreads 开销** | tiled kernel 内有 2 × (N/TILE) = 128 次 syncthreads，每次线程屏障都有数百 ns 开销 | 次要 |
| **共享内存 bank 配置开销** | TILE=16 时 shared memory 读写布局未必全无 bank conflict | 边际 |
| **naive 算法本身的 ILP** | 编译器把 inner-loop k 做了指令级并行，naive 的 effective throughput 高于教学示意 | 边际 |

### 3.2 如何让 tiled 优势真正显出来？

| 改动 | 预期效果 |
|---|---|
| 把 N 从 1024 调到 4096 | naive 访存不再全在 L2 命中，加速比应该跳到 4-8× |
| 把 N 调到 8192+ | naive 几乎全走 HBM，加速比接近 TILE 上限 16× |
| 调用 `cudaMemAdvise(d_A, bytes, cudaMemAdviseSetPreferredLocation, 0)` | 显式控制 residency，把 L2 影响剥离 |
| 改用 `ncu` 测 L2 hit rate 验证假说 | 直接证据 —— 见 phase2 |

**教学含义**：教学 lab 在 N=1024 跑 matmul，**很容易让学生以为"tiled 只是稍好一点的优化"**——这是个错觉。要让学生真正体会 tiled 价值，N 至少应到 2048 或更大，让数据超出 L2 才能凸显 naive 的访存压力。

### 3.3 与 cuBLAS 横向对比

| 实现 | 1024×1024 SGEMM 耗时 | 相对最快 |
|---|---|---|
| `matmul_naive`（本练习） | 686.6 µs | 7.38× |
| `matmul_tiled`（本练习） | 451.8 µs | 4.85× |
| `cublasSgemm`（exercise 08） | 93.0 µs | 1.00× |

cuBLAS 比 tiled 快 **4.85×**——比 tiled 相对 naive 的 1.52× 大得多。说明工程级优化（Tensor Core、tilesize 选型、双缓冲、寄存器分块、warp-level primitives 等）的收益远大于"教学示意上的 tiled 优化"。**自己写的库代码大概率打不过 cuBLAS，工程上首选是直接调库。**

## 4. 思考延伸

1. 把 N 从 1024 改成 4096，重跑 profile.sh，对比加速比变化。预期应明显 > 1.52×。
2. 把 TILE 改成 32（同时把 block 改成 32×32=1024 线程），是否加速比提升？注意 shared memory 是否够（32×32×4B×2=8 KB/block，H20 每 SM 228 KB，可驻 28+ block）。
3. 用 `ncu --section MemoryWorkloadAnalysis` 看两种 kernel 的 L2 hit rate，验证本分析"主要失分在 L2"的假说。
4. 既然 `cudaLaunchKernel` 自身耗时 15-122 µs，单 kernel 程序到底跑多短的 kernel 才会让 launch overhead 主导？（→ 见 exercise 06 实测）

## 附录：复现命令

```bash
cd phase1_nsys/02_matmul && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum    --format csv ./report02.nsys-rep
$NSYS stats --force-export=true --report cuda_kern_exec_sum   --format csv ./report02.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report02.nsys-rep
```
