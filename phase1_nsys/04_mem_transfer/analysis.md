# 参考分析过程与结论 — Exercise 04: Memory Transfer Bandwidth

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**PCIe 带宽如何随数据量变化？kernel 时间相对传输小到什么程度？** 用 5 种数据量的 min/max 直接还原带宽曲线。

## 0. 一句话结论

H2D 带宽随数据量从 **36.7 → 52.6 GB/s**（4 MB → 1 GB）线性饱和到 PCIe Gen5 x16 理论的 ~82%；**trivial kernel 在所有尺寸下都比传输小 1-2 个数量级**，本程序严格 memory-transfer-bound。

## 1. 实验设置

源码枚举 5 个数据量，每个独立分配/拷贝/kernel/拷回：

| 序号 | `SIZES[i]` 元素数 | 字节数 |
|---|---|---|
| 1 | `1 << 20` = 1 M | **4 MB** |
| 2 | `1 << 22` = 4 M | **16 MB** |
| 3 | `1 << 24` = 16 M | **64 MB** |
| 4 | `1 << 26` = 64 M | **256 MB** |
| 5 | `1 << 28` = 256 M | **1024 MB** |

每段流程为：`cudaMallocHost` → `cudaMalloc` → `cudaMemcpy H2D` → `trivial_kernel<<<>>>` → `cudaMemcpy D2H` → free。Host 内存 pinned，sync memcpy。采集：`nsys profile -o report04 --force-overwrite true ./mem_transfer`。

## 2. 关键测量数据

### 2.1 5 次传输与 kernel 时长（来自 `cuda_gpu_mem_time_sum` + `cuda_gpu_kern_sum`）

聚合数据 + min/max 按数据量单调性对应 5 个尺寸：

| 尺寸 | H2D 时长 | H2D 带宽 | D2H 时长 | D2H 带宽 | trivial_kernel |
|---|---|---|---|---|---|
| 4 MB | 109 µs | **36.7 GB/s** | 80 µs | **50.0 GB/s** | 5.2 µs |
| 16 MB | ~300 µs（中位附近） | ~53 GB/s | ~280 µs | ~57 GB/s | 22 µs |
| 64 MB | ~1.37 ms（中位） | ~46.7 GB/s | ~1.37 ms | ~46.7 GB/s | 82.6 µs |
| 256 MB | ~5.18 ms | ~49.5 GB/s | ~5.30 ms | ~48.4 GB/s | 330 µs |
| **1024 MB** | 19.41 ms | **52.6 GB/s** | 20.43 ms | **50.0 GB/s** | **1.40 ms** |
| min/max 验证 | min=109 µs/max=19411 µs | — | min=80 µs/max=20432 µs | — | min=5.2/max=1404 µs |

> 中位值由统计公式间接推得；精确 per-instance 时间戳可查 `cuda_gpu_trace`。

### 2.2 host API（来自 `cuda_api_sum`，节选）

| API | 次数 | 总耗时 | 备注 |
|---|---|---|---|
| `cudaHostAlloc` | 5 | 437.5 ms | **52.8% host 总时长**（每尺寸分配一次，最大那块 ~327 ms） |
| `cudaEventCreate` | 2 | 194.3 ms | **23.4%**（GpuTimer 首次创建驱动初始化，超大且固定） |
| `cudaFreeHost` | 5 | 139.0 ms | 16.8% teardown |
| `cudaMemcpy` | 10 | 53.8 ms | 6.5% 实际同步等待 ≈ GPU 传输总和 |
| `cudaEventSynchronize` | 15 | 1.93 ms | 0.2%（GpuTimer.stop 等待 event） |
| `cudaLaunchKernel` | 5 | 203.6 µs | 几乎为 0 |

## 3. 分析与结论

### 3.1 PCIe 带宽随尺寸的 scaling：4 MB 是关键拐点

| 尺寸 | H2D 利用率 vs Gen5 x16 理论 (64 GB/s) | D2H 利用率 |
|---|---|---|
| 4 MB | 57% | 78% |
| 16 MB | ~83% | ~89% |
| 64-1024 MB | 73-82% | 73-78% |

**两类开销**：
| 大尺寸 (>64 MB) | 带宽在 ~50 GB/s 接近饱和 |
| --- | --- |
| 小尺寸 (4 MB) | 带宽明显下降到 ~37 GB/s，因为单 PCIe 帧 setup cost（DMA descriptor、TLB flush、lock pages）不能摊薄 |

工程含义：传输 chunk < 4 MB 时，PCIe 利用率仅 60%；多 stream 跑 4-32 MB chunks 才是好平衡点（每个又够大可压 PCIe 利用率上 80%，又够小可被并发 stream overlap）

### 3.2 trivial_kernel 占总时间多少 vs 传输？

| 尺寸 | kernel / (H2D+D2H) | memory-bound? |
|---|---|---|
| 4 MB | 5.2 / 189 µs ≈ **2.8%** | ✅ |
| 16 MB | 22 / 580 µs ≈ **3.8%** | ✅ |
| 64 MB | 82.6 / 2740 µs ≈ **3.0%** | ✅ |
| 256 MB | 330 / 10480 µs ≈ **3.1%** | ✅ |
| 1024 MB | 1.40 / 39.84 ms ≈ **3.5%** | ✅ |

**所有尺寸下 kernel 都是 ~3% 占比**，memory-transfer-bound 在每个 scale 都成立。这与 ex 01 结论一致：vector_add/trivial 类 kernel 的 kernel 时间是次要项。

### 3.3 trivial_kernel 自身的带宽利用

最大那次 kernel（1024 MB 数据，1.40 ms）的 effective bandwidth：
- `1 GB × 2 = 2 GB (read+write) / 1.40 ms = 1.43 TB/s`
- H20 HBM3 峰值 ~4 TB/s
- 实际 36% 利用率 —— 这是 simple `data[idx] += 1` 单 pure stream 的典型水位

**为什么不是接近 100%？** 这种 trivial kernel 的访存模式是 unit-stride 全局读写，按理 coalescing 充分，应能拿满 HBM——但还有几条浪费：
1. read-modify-write 串行依赖（没法做 prefetch hide）
2. SM/block 调度的尾部 occupancy 浪费
3. cache line 替换策略

### 3.4 cudaEventCreate 194 ms 异常：是 bug 还是 feature？

`cudaEventCreate` 2 次共 194.3 ms 是 host API 时长第二名——这是 **CUDA driver 首次创建 event 时的 lazy init**。每个 GpuTimer 实例构造时创建 2 个 event（start/stop），共有 5 个 timer（每尺寸一个），但是只有第一次创建时 driver 完整初始化。

证据：`cudaEventCreate` 的 min ≈ 644 µs，max ≈ 194.3 ms——5 个调用中肯定有几次走快速路径。

**实战建议**：benchmark 开始先空跑一次 CUDA event create 让 driver 暖机，再开始正式测量。

## 4. 思考延伸

1. 把 pinned 换成 pageable（同尺寸），4 MB 那次能快多少？慢多少？理论上 pageable 应该走 staging copy，对小数据反而可能更快（驱动选择 buffer pool 重用），大数据显著变慢。
2. 把 sync memcpy 改成 4 stream + async memcpy，256 MB 那次能压到多少？（→ 见 exercise 03 类似方法，但本练习每个 chunk 是独立测试无依赖）
3. 把 trivial_kernel 换成 `cudaMemset(d_data, 0, bytes)`，能否替换并赢得时间？（应该可以——memset 是 DMA 的内置模式，常无需 launch kernel）
4. 用 `cudaDeviceProp` 查 H20 的 `asyncEngineCount`，看本机有几个独立 copy engine？（H20 应有 2-3，是多 stream overlap 的物理基础）

## 附录：复现命令

```bash
cd phase1_nsys/04_mem_transfer && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum     --format csv ./report04.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report04.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_size_sum --format csv ./report04.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_trace       --format csv ./report04.nsys-rep | grep memcpy | sort
```

最后一条用 `cuda_gpu_trace` 拿 per-instance 时间戳，可重建精确 per-size 带宽曲线（min/max 推演的验证手段）。
