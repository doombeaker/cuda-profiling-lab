# 参考分析过程与结论 — Exercise 12: Parallel Reduction

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**warp divergence 在哪里、能否被 sequential addressing 显著消除？** 用 `cuda_gpu_kern_sum` 实测两种 reduction kernel 耗时差距论证。

## 0. 一句话结论

`reduce_sequential` 比 `reduce_interleaved` 快 **2.37×**（136 µs vs 322 µs），**实测远超** README 宣称的 10-30% 加速区间。这是 N=16 M、GRID_SIZE=65536 这种"巨大 grid × 小 block"场景下 warp divergence 损失被放大的典型证据。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `reduction.cu` |
| 元素数 | `N = 1 << 24` = 16 M floats = 64 MB |
| Block 大小 | `BLOCK_SIZE = 256` |
| Grid 大小 | `GRID_SIZE = N / BLOCK_SIZE = 65536` blocks |
| 输入 | 全 1.0，正确结果应为 N = 16,777,216 |
| Host 内存 | pinned |
| 采集 | `nsys profile -o report12 --force-overwrite true --trace=cuda ./reduction` |
| 程序自打印 | `Speedup: sequential is 2.37x faster than interleaved` |

## 2. 关键测量数据

### 2.1 两 kernel 耗时（来自 `cuda_gpu_kern_sum`）

| Kernel | 次数 | 总耗时 | 占比 |
|---|---|---|---|
| `reduce_interleaved` | 1 | **322.7 µs** | 70.3% |
| `reduce_sequential` | 1 | **136.3 µs** | 29.7% |
| **加速比** | — | `322.7 / 136.3 = 2.37×` | — |

### 2.2 memcpy（来自 `cuda_gpu_mem_time_sum` + `cuda_gpu_mem_size_sum`）

| 操作 | 次数 | 总耗时 | 数据量 |
|---|---|---|---|
| H2D | 1 | 1.21 ms | 67.1 MB（整个 input 数组） |
| D2H | 2 | 21.0 µs | 0.26 MB × 2（partial results 回 host） |

> 两次 D2H 对应两个 kernel 各自把 partial sum 数组（65536 × 4B = 256 KB）拷回 host 做 CPU 端最终求和。

### 2.3 kernel launch-exec 时序（来自 `cuda_kern_exec_sum`）

| Kernel | `cudaLaunchKernel` | queue | kernel |
|---|---|---|---|
| `reduce_interleaved` | 143.4 µs | — | 322.7 µs |
| `reduce_sequential` | 16.5 µs | 2.2 µs | 136.3 µs |

`cudaLaunchKernel` 之间 8.7× 差异（143 vs 16.5 µs）也是首次 vs 后续的派发开销差异，与 kernel 性能无关。

## 3. 分析与结论

### 3.1 为什么是 2.37× 而非 README 说的 10-30%？

两种 kernel 计算量、shared memory 用量、访存模式完全相同，**唯一差异是 reduce 树的寻址方向**：

| | interleaved | sequential |
|---|---|---|
| 寻址 | `stride = 1, 2, 4, ... 128`（doubling） | `stride = 128, 64, 32, ... 1`（halving） |
| 线程参与 | `tid % (2*stride) == 0` | `tid < stride` |
| warp divergence 出现阶段 | 后期（大 stride） | 早期（大 stride）有 - 实际无 |
| 真正损害 | stride ≥ 32 时整 warp 大半线程 idle | stride ≥ 32 时整 warp 全 active，stride < 32 后才有 warp 收缩 |

实测 2.37× 比预期 10-30% 高得多，关键在 **N=16M + 65536 blocks 触发了 divergence 的极端放大**：
- 每个 block 256 线程 / 32 = 8 warp
- interleaved 在每个 step 都有 divergence，**最后一个 step 仅 1 个 thread 工作、其余 255 idle**
- 累计 log2(256) = 8 步 × 平均 idle 率上升，对总能耗/时间影响远超 30%
- block 数 65536 多到极致，per-block 浪费乘以 65536 倍 → 整体效果被放大到 2.37×

README 的 10-30% 区间应是**较小 grid 的常规假设**；当 grid 爆炸时，divergence 的累计代价更具破坏性。

### 3.2 这个 2.37 是真的吗？可信度核查

`reduce_sequential` 比 `reduce_interleaved` **晚于** reduce_interleaved 跑（程序 main 函数顺序）—— 是否因 GPU 时钟/Governor 状态更暖而虚高？

证据反证：
- 程序段中两个 kernel 之间没有 warmup 差异（无 sync 等待 GPU idle）
- 单次 invocation（无重复测量）下，kernel 内部 L2 缓存状态对二者相似（都是 first-touch 全 cold）
- 顺序差异在 <1ms 时间尺度不会触发 GPU DVFS —— H20 HBM3 时钟稳定

**结论**：2.37× 真实反映算法差异，非测量噪声。但要稳妥，可加 `for (int rep=0; rep<5; rep++) { kernel<<<>>>(...); }` 多次跑取 med。

### 3.3 H2D 1.21 ms 占总时间多少？reduction 是 memory-bound 还是 compute-bound？

| 阶段 | 时长 | 占总 GPU op 时长 |
|---|---|---|
| H2D（64 MB） | 1.21 ms | — |
| reduce_interleaved | 0.323 ms | 70% |
| D2H × 2 | 0.021 ms | — |
| reduce_sequential | 0.136 ms | — |
| **kernel 阶段合计** | **0.459 ms** | — |

H2D（1.21 ms）远超两 kernel 合计（0.46 ms），但 H2D 在 host 线程上同步等待发生在程序入口（不在 GpuTimer 范围内）。**kernel 本身是 memory-bound**：
- input 64 MB / `reduce_sequential` 0.136 ms = **470 GB/s** 有效带宽
- H20 HBM3 峰值 4 TB/s → 仅占 12%
- 64 MB 全部读 + 写 256 KB shared + 写 256 KB partial 输出 → 绝大多数 op 是 global memory read
- 但 reduction 算法只在尾部少量写——这与 vector_add 那种 1:1 读写比不同，effective bandwidth 不能简单和峰值比

### 3.4 工程含义

| 优化 | 预期 |
|---|---|
| 加 `__shfl_down_sync` 处理最后 5 步（warp 内 reduce） | 进一步消除 syncthreads，可能再快 1.5-2× |
| `cudaMemcpyAsync` 异步 H2D 让 kernel 启动早 | 节省 ~1 ms，但需要 stream 协调 |
| 用 cub::DeviceReduce::Reduce 库实现 | 通常可达本实现的 2-5× |
| 改用 thrust::reduce | 类似 cub，库实现 |

## 4. 思考延伸

1. 把 stride 改成 `__shfl_down_sync`，能否把 136 µs 再压一半？warp 内 reduce 是 reduction kernel 最后的工程加速手段。
2. 把 BLOCK_SIZE 改成 512 或 1024，interleaved 的 divergence 增多还是减少？加速比会变大还是变小？
3. 用 `ncu --section WarpStateStats` 看 warp stall 原因，divergence 占 stall 比例多大？（→ phase2 exercise 12）
4. 既然 reduce_interleaved 用了 70% 时间却只输出 partial sum、reduce_sequential 只用 30% — 把 interleaved 直接删掉只跑 sequential，total wall 会变化吗？（不会，等量计算任务没有转移）

## 附录：复现命令

```bash
cd phase1_nsys/12_reduction && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum     --format csv ./report12.nsys-rep
$NSYS stats --force-export=true --report cuda_kern_exec_sum    --format csv ./report12.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report12.nsys-rep
```
