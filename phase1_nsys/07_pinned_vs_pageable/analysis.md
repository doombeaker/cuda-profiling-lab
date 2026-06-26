# 参考分析过程与结论 — Exercise 07: Pinned vs Pageable Memory

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**pinned 内存到底比 pageable 快多少？pageable 的 staging copy 在 nsys 数据里能不能看到？**

## 0. 一句话结论

pinned 单次 H2D 实测 **4.87 ms / 52.3 GB/s**，pageable **26.89 ms / 9.5 GB/s**——**pinned 比 pageable 快 5.5×**，正好处于 README 区间 2-10× 中段。256 MB 在 H20 HBM3 上对 PCIe Gen5 x16 利用率：pinned 82%、pageable 15%（被 staging copy 拖累）。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `pinned_vs_pageable.cu` |
| 测试数据量 | `TEST_SIZE = 1 << 26` = 64 M float = **256 MB** |
| 三组测试 | (1) pinned `cudaMallocHost` + H2D；(2) pageable `malloc` + H2D + 一次 trivial_kernel；(3) 对照 summary：pinned vs pageable 重测 |
| 采集 | `nsys profile -o report07 --force-overwrite true ./pinned_vs_pageable` |
| 自打印（节选） | 通过 cuda_event 测出 pinned H2D 时长 vs pageable H2D 时长 |

## 2. 关键测量数据

### 2.1 4 次 H2D memcpy（来自 `cuda_gpu_mem_time_sum` + `cuda_gpu_mem_size_sum`）

| 操作 | 次数 | 总耗时 | 单次 avg | Min | Max |
|---|---|---|---|---|---|
| H2D memcpy | 4 | 63.4 ms | 15.85 ms | **4.867 ms** | **26.888 ms** |
| H2D 数据量 | 4 | 1073.7 MB | 268.4 MB | 268.4 MB | 268.4 MB |

4 次中 2 次为 pinned、2 次为 pageable。结合 README 期望与 min/max：
- **Pinned H2D** × 2：min ~4.867 ms，对应带宽 256 MB / 4.867 ms = **52.6 GB/s**
- **Pageable H2D** × 2：max ~26.888 ms，对应带宽 256 MB / 26.888 ms = **9.52 GB/s**
- **Speedup = 5.53×** ✅ 在 README "2-10×" 区间中段

### 2.2 trivial_kernel（pageable 数据上跑）

| Kernel | 次数 | 总耗时 | 单次平均 |
|---|---|---|---|
| `trivial_kernel` | 1 | 0.345 ms | 0.345 ms（256 MB 数据上 +1.0 iter） |

与 ex 04 的 256 MB kernel (330 µs) 几乎相同——kernel 自身耗时与 host memory 类型无关（device 内存里的数据已被 staging 复制过去）。

### 2.3 host API（节选自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 备注 |
|---|---|---|---|
| `cudaEventCreate` | 2 | 197.0 ms | 40.2%，**GpuTimer 首次创建驱动初始化**（同 ex 04 异常） |
| `cudaHostAlloc` | 2 | 169.6 ms | 34.6%，pinned 内存一次性分配 |
| `cudaMemcpy` | 4 | 63.7 ms | 13.0%，实际同步等待 GPU 完成 |
| `cudaFreeHost` | 2 | 57.0 ms | 11.6%，pinned 内存释放 |
| `cudaEventSynchronize` | 5 | 0.40 ms | 0.2% |
| `cudaLaunchKernel` | 1 | 0.137 ms | <0.1% |

## 3. 分析与结论

### 3.1 5.5× 加速从何而来？staging copy 的 2× 工作量数学

Pageable H2D 内部走的两步：
```
[host pageable buffer] ──memcpy──> [pinned staging buffer] ──DMA──> [device HBM]
                              ↑ CPU 工作                              ↑ GPU DMA
```

每个字节被搬了 **2 次**：先 CPU 在物理内存间 memcpy（CPU 工作），再由 DMA 推到 GPU。两次搬移的实测时间加起来：

| 步骤 | 单次 256 MB 耗时 |
|---|---|
| CPU memcpy（pageable → staging） | ~10 ms（按 host memcpy ~25 GB/s 估算） |
| DMA（staging → device） | ~5 ms（与 pinned 直 DMA 同速） |
| **合计** | **~15 ms 预测值 vs 实测 26.9 ms** |

实测比预测更高。差异来自：
- **cudaMallocHost 创建 staging buffer 是 driver 内部行为**，每次 cudaMemcpy(pageable) 都临时分配/释放 staging —— 这本身有 host-side 开销
- **staging buffer 不一定在物理上连续**，DMA 性能可能打折
- **page fault on first-time access host buffer**：pageable 的虚拟页可能未映射到物理页，driver 第一次访问触发 page fault

Pinned H2D 跳过这些步骤：DMA 直接读 pinned 物理页。**5.5× 不是理论上限——理论上限是无穷大（如果 CPU memcpy 完全 serialize 阻塞 DMA）。**

### 3.2 "staging copy 是 2 个 H2D" 在 nsys 数据里能直接看到吗？

理论上 nsys 应能显示 2 个独立的 memcpy 操作（一次 staging → 一次实际 DMA），但**实测 cuda_gpu_mem_time_sum 只看到 4 个 H2D 操作（每种 2 次）**，没有额外的 staging memcpy 事件。

原因：staging copy 发生在 driver 内部，**driver 不通过 CUDA API 报告它**——CUPTI 只看到"用户层面"这次 cudaMemcpy。要做细致 staging 分析，得用：
- **`ncu` profile + `--section MemoryWorkloadAnalysis`** —— 看 PCIe 真实带宽利用率
- **`nvprof --metrics pci_elapsed_clocks`** —— 但 nvprof 已 deprecated
- **PMU sampling** —— 通过 `perf` 抓 driver 内存复制指令

对教学目的，**nsys 提供"宏观时长对比"已够**；想看 staging 拓扑需 ncu。

### 3.3 教学含义：何时选 pinned 何时不选

| 场景 | 选择 | 原因 |
|---|---|---|
| 高频 H2D/D2H（如训练 batch 流式传输） | **pinned** | 5.5× 带宽提升，远超 setup 成本摊销 |
| 一次性大数据传输（如启动加载权重） | **pinned** | 单次也能拿 5× 加速 |
| 短生命周期 + 一次性 malloc/free（如 microbenchmark 测单次 memcpy） | **可 pageable** | pinned 一次性 setup 成本（~85 ms）超出 pageable 慢 22 ms 的代价 |
| 不能锁住太多物理内存（系统 16 GB+ 占用 pinned） | **pageable 混用** | pinned 锁住物理 RAM，过量导致 OS OOM 风险 |
| 与 CPU 用户态代码共享同一 buffer（如 zero-copy RDMA） | **看架构** | H20 + GPUDirect RDMA 可走 pinned |

### 3.4 cudaEventCreate 197 ms 又出现了

`cudaEventCreate` 2 次总 197 ms（同 ex 04 显著），印证这是 **CUDA 在 driver 第一次被启用时 lazy-init 的成本**。所有用到 GpuTimer 的练习（ex 04, ex 07 都有）都会看到这个开销。

教学提醒：**任何 CUDA benchmark 第一个 GpuTimer 实例创建时都会付这个代价**。生产代码或 benchmark 框架应：
1. 启动时先创建一个 dummy event 让 driver 暖机
2. 正式测量从第二个 GpuTimer 实例开始

## 4. 思考延伸

1. 把测试数据量从 256 MB 改成 4 MB（小数据），pinned vs pageable speedup 应该会变小（setup 占比上升）。验证实验：
   ```c++
   constexpr int TEST_SIZE = 1 << 20;  // 4 MB
   ```
2. 用 `cudaHostRegister(pageable_ptr, size, cudaHostRegisterDefault)` 把已有 pageable 内存**注册**为 pinned —— 既能拿 pinned 速度又不需要 `cudaMallocHost` 重新分配，setup 成本应下降。
3. `cudaMemcpyAsync(pageable, ...)` 配合 stream，driver 会内部串行化同步问题——加 `cudaStreamSynchronize` 看是否拿到 pinned 一半带宽？
4. 用 `strace -e clone,mmap,munmap -f ./pinned_vs_pageable 2>&1 | grep -c mmap` 数一下两种模式各调了多少次 mmap，间接量化 staging buffer 分配次数。

## 附录：复现命令

```bash
cd phase1_nsys/07_pinned_vs_pageable && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report07.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_size_sum --format csv ./report07.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_trace       --format csv ./report07.nsys-rep | grep memcpy | sort -k2 -n  # 排序看 min/max 各对应哪种
```
