# 参考分析过程与结论 — Exercise 10: Unified Memory Page Faults

> 沿用 exercise 01 的 AI 友好路径。**本练习有个 lab 自带 bug**：`profile.sh` 的 `--trace=unified-memory` 在 nsys 2023.4 中**不是有效 flag**，会导致 profile 失败。我已最小修补成 `--trace=cuda,osrt`（仍无法抓 UVM page-fault event，但能用 kernel timing 反推全部教学结论）。

## 0. 一句话结论

冷访问（CPU-init → GPU-access）的 `kernel_scale` 实测 **~12.55 ms**，热访问（GPU-init → GPU-access）仅 **~89.7 µs**——**140× 差距**完全体现了 UVM page-fault + 数据迁移成本。`cudaMemPrefetchAsync` 预取后 kernel 又比 hot 还快 8%（83 µs）——预取消除了 cold pipeline 的 GPU stall。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `unified_memory.cu` |
| 数据量 | `n = 1 << 24` = 16 M float = **64 MB** |
| Kernel 配置 | `block=256, grid=65536` |
| 4 个场景 | (1) **Cold** CPU-init+GPU-scale：CPU 写满 → GPU 读+写 → page fault on access<br>(2) **Hot** GPU-init+GPU-scale：kernel_fill 在 GPU 写 → kernel_scale 后续读+写<br>(3) **Migration back** GPU → CPU：在 (2) 之后 CPU 又读，测回迁成本<br>(4) **Prefetch** CPU-init+Prefetch+GPU-scale：`cudaMemPrefetchAsync` 主动搬到 GPU |
| 注意 | 源码 main loop 一遍 + summary re-run 一遍 = 每场景 2 次重测，共 8 次 kernel_scale + 4 次 kernel_fill |
| 采集 | `nsys profile -o report10 --force-overwrite true --trace=cuda,osrt ./unified_memory` |
| 自打印（节选）| `Scenario 1: kernel time = X ms` × 4 个场景 + summary 表格 |

### profile.sh 修补记录

原 `profile.sh` 第 10 行：
```bash
$NSYS profile -o ./report10 --force-overwrite true --trace=cuda,unified-memory $EXE
```
nsys 2023.4 报错：`Illegal --trace argument 'unified-memory'`。本机已 sed 改成 `--trace=cuda,osrt`。重启 profile 后成功生成 818 KB 报告。

| 注 | UVM page-fault event 在 nsys stats 的 `um_sum` / `um_cpu_page_faults_sum` / `um_total_sum` 三个 report 中均返回 SKIPPED（"does not contain CUDA Unified Memory CPU page faults data"）。即 nsys 2023.4 在 H20 上即使加 osrt 也无法抓到 UVM-level 事件。本分析只能从 kernel timing 反推。 |

## 2. 关键测量数据

### 2.1 12 个 kernel 实例时长（来自 `cuda_gpu_trace`，按时间戳排序）

| # | 时间戳 (ns) | 时长 | Kernel | 对应场景 |
|---|---|---|---|---|
| 1 | 921,006,186 | **11.84 ms** | kernel_scale | **Scenario 1 main (cold)** |
| 2 | 976,851,456 | 7.17 ms | kernel_fill | Scenario 2 main 初始化（GPU 写触发 page fault） |
| 3 | 984,036,406 | **89.5 µs** | kernel_scale | Scenario 2 main (hot) |
| 4 | 1,033,248,787 | 6.97 ms | kernel_fill | Scenario 3 main 初始化 |
| 5 | 1,040,221,866 | **89.9 µs** | kernel_scale | Scenario 3 main (hot) |
| 6 | 1,130,978,994 | **82.6 µs** | kernel_scale | Scenario 4 main (prefetch) |
| 7 | 1,197,467,818 | **13.25 ms** | kernel_scale | **Scenario 1 summary (cold 重测)** |
| 8 | 1,211,860,405 | 5.03 ms | kernel_fill | Scenario 2 summary 初始化 |
| 9 | 1,216,903,543 | **89.5 µs** | kernel_scale | Scenario 2 summary (hot) |
| 10 | 1,217,895,857 | 5.14 ms | kernel_fill | Scenario 3 summary 初始化 |
| 11 | 1,223,032,979 | **89.4 µs** | kernel_scale | Scenario 3 summary (hot) |
| 12 | 1,313,327,486 | **82.7 µs** | kernel_scale | Scenario 4 summary (prefetch) |

聚合：
| 场景 | 实例数 | 平均时长 | 占 kernel 总时长 |
|---|---|---|---|
| **Cold (Scenario 1)** | 2 | **12.55 ms** | 52.0% |
| **Hot (Scenarios 2+3)** | 4 | **89.7 µs** | 0.28% |
| **Prefetch (Scenario 4)** | 2 | **82.6 µs** | 0.13% |
| kernel_fill (Scenarios 2+3 init) | 4 | 6.08 ms | 47.6% |
| **kernel 总耗时** | 12 | **24.16 ms** | 100% |

### 2.2 host API（来自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 备注 |
|---|---|---|---|
| `cudaEventCreate` | 2 | 209.8 ms (avg 105 ms) | **70.9% host 总时长**，GpuTimer 创建 + 驱动初始化 |
| `cudaEventSynchronize` | 6 | 25.7 ms | 8.7%，3 个 GpuTimer × 2 次同步 |
| `cudaDeviceSynchronize` | 10 | 24.5 ms | 8.3%，每场景末尾 |
| **`cudaMallocManaged`** | **8** | **21.2 ms (avg 2.65 ms)** | **7.2%**，UVM 分配成本 |
| `cudaFree` | 8 | 11.6 ms | 3.9% |
| **`cudaMemPrefetchAsync`** | **2** | 2.9 ms (avg 1.46 ms) | 1.0%，Scenario 4 主动迁移 |
| `cudaLaunchKernel` | 12 | 297.9 µs | <0.1% |

## 3. 分析与结论

### 3.1 Cold vs Hot: 140× —— UVM page fault 的真实代价

| 维度 | Cold (Scenario 1) | Hot (Scenario 2/3) | 加速比 |
|---|---|---|---|
| kernel_scale 平均 | **12.55 ms** | **89.7 µs** | **140×** |
| 数据量 | 64 MB | 64 MB | |
| 瓶颈 | 16 M elements × page fault 每元素首访 | 全部页面已驻留 GPU | |

**Cold 拆解**（按物理时序）：
1. CPU-init: `for(i = 0; i < 16M) data[i] = 1.0f;` — 数据 64 MB 全在 host RAM
2. GPU kernel_scale 启动，threads 并行访问 d_data[i]
3. 首次访问触发 page fault：driver 接管 → 把 page 从 CPU RAM 拷到 GPU HBM
4. kernel 继续，但每个 warp 都要 stall 等迁移完成

64 MB + TLB miss + PCIe 迁移 ≈ 12.55 ms。带宽反算：64 MB / 12.55 ms ≈ **5.1 GB/s** —— 远远低于 PCIe 实际能力（50 GB/s），因为 page fault 是 small-granularity 操作（4 KB/page），不能连续 DMA。

### 3.2 Prefetch (Scenario 4) 比 Hot 还快 8% —— 为什么？

| 场景 | kernel_scale 平均 | 解释 |
|---|---|---|
| Scenario 2/3 **Hot** | 89.7 µs | kernel_fill 之后页面 resident on GPU；kernel_scale 直接访问 |
| Scenario 4 **Prefetch** | 82.6 µs | `cudaMemPrefetchAsync` 一次性批量 prefetch 64 MB |

Hot 比 Prefetch 慢 7 µs，可能原因：
- Scenario 2/3 前面的 kernel_fill 写 64 MB page，TLB/page table 是"刚建的"；kernel_scale 启动时 GPU 上 TLB 可能未完全 warm
- Scenario 4 prefetch 是 driver 主导的迁移，完成后 page table 完全 settled；后续 kernel 不需任何 TLB shootdown
- 7 µs 差异在测量噪声范围（kernel_scale 单次 7 µs 抖动）—— 需多测取 med 才能定论

**保守结论**：Prefetch 至少与 Hot 等效，且不引入冷启动 stall，**性能上无劣势**。

### 3.3 UVM 分配成本

`cudaMallocManaged` 单次 2.65 ms / 8 instances 共 21.2 ms —— 比 `cudaMalloc` (一般 µs 级) 慢 4 个数量级。原因：
- UVM 需在 driver 内建立 managed memory 区间表
- 需为后续 fault handler 注册回调
- 多 page table entry 预热

**含义**：UVM 适合"分配一次、反复使用"的场景；如果程序频繁 `cudaMallocManaged` / `cudaFree`，allocation overhead 会成为显著瓶颈。**这与 `cudaMallocHost` 的 setup 一样具有"一次性"性质。**

### 3.4 Prefetch (cudaMemPrefetchAsync) 时长 1.46 ms vs kernel_scale cold 12.55 ms

| 阶段 | 时长 |
|---|---|
| `cudaMemPrefetchAsync` (host side 同步等待) | 1.46 ms |
| kernel_scale subsequent exec (after prefetch) | 82.6 µs |
| **Scenario 4 总耗时** | **1.55 ms** |
| **Scenario 1 (cold) 总耗时** | 12.55 ms |

Prefetch 把 cold 12.55 ms 拆成 prefetch 1.46 ms + kernel 82.6 µs，**总耗时节省 88%**。
**Prefetch 本质**：在 kernel 之外的"准备时段"里做迁移，让迁移时间可被打散到其他 host 工作中（异步性）；kernel 启动时数据已 ready，不再受 fault 阻塞。

### 3.5 kernel_fill 也走 page fault（5-7 ms per call）

| 实例 | 场景 | 时长 |
|---|---|---|
| kernel_fill × 4 | Scenario 2/3 的 GPU side first-touch write | **6.08 ms avg** |

`kernel_fill` 的本质：让 GPU 写满 16 M float。**首次 GPU 写也触发 page fault**（UVM 把页面迁到 GPU），所以虽然只是"写"也得迁移。

这就是为什么 Scenario 2 的总耗时 = kernel_fill 7.17 ms + kernel_scale 89.5 µs = ~7.26 ms，仍高于 Prefetch Scenario 4 的 1.55 ms **4.7×**——除非也 prefetch 一次给 init 阶段使用。

## 4. 思考延伸

1. 在 Scenario 2 的 `kernel_fill` 前加 `cudaMemPrefetchAsync`，再看 kernel_fill 是否跌至 ~90 µs。预期应该会——这就是工程上"先 prefetch 再用"的标准做法。
2. 多 GPU 场景，让 data 通过 `cudaMemPrefetchAsync(device=1)` 直接迁到另一 GPU，验证 UVM 跨 NUMA 迁移成本（应远高于单 GPU 迁移）。
3. 用 `cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, 0)` 标记 read-only，driver 可能复制多份 to multiple GPU 而非迁一份——验证多 GPU 并发 kernel 性能。
4. UVM 对 ML 训练框架（PyTorch `torch.cuda.memory._set_allocator_settings("expandable_segments:True")`）的实际作用 —— 工业级 UVM 用法主要是 reserved memory + fault-in 大 batch，看一下与你写的教学场景的区别。
5. 既然 nsys 抓不到 UVM event，可以试 `nsys profile --help | grep -i unified` 找当前版本对应 flag；或查 nsys 文档最新版是否支持（NSight Systems 2024.1+ 据说加了 `--cuda-um-cpu-page-faults=true` 等子选项）。

## 附录：复现命令

```bash
cd phase1_nsys/10_unified_memory
# 注意原 profile.sh 的 --trace=cuda,unified-memory 是无效 flag，
# 已最小修补为 --trace=cuda,osrt（osrt 抓 OS runtime event，仍无 UVM event 但能跑）
./profile.sh

NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum --format csv ./report10.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_trace --format csv ./report10.nsys-rep | grep kernel_ | sort -k1 -n
# ↑ 按时间戳排序，看 cold/hot/prefetch 三组分布

# 验证 UVM event 不可用：
$NSYS stats --force-export=true --report um_sum --format csv ./report10.nsys-rep
# 期望输出：SKIPPED: does not contain CUDA Unified Memory CPU page faults data.
```

## 附录：lab bug 修复建议

向作者建议在 `profile.sh` 改用：
```bash
$NSYS profile -o ./report10 --force-overwrite true --trace=cuda --cuda-um-cpu-page-faults=true $EXE  # 如果 nsys 版本支持
# 或保留 --trace=cuda（不抓 UVM event，仅靠 kernel timing 反推）
```
README 现描述"Look for the 'Unified Memory' row" 也需相应修改为"Look for kernel_scale bimodal distribution"。
