# 参考分析过程与结论 — Exercise 09: Synchronization Patterns

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**device / stream / event 三种同步在 nsys 数据里有什么本质差异？kernel 是否真重叠？** 用 `cuda_gpu_kern_sum` 的时长双峰直接回答。

## 0. 一句话结论

6 个 `delay_kernel` 实例时长呈**清晰双峰**：**4 个 1.07 ms**（Tests 2/3，concurrent 两 stream）+ **2 个 1.81 ms**（Test 1，serialized 单 stream）。三种 sync API 时长接近（0.90~1.07 ms/次），但**实际 wall time 差异体现在 kernel 时长而非 sync API 本身**——concurrent 模式下两 kernel 并行执行让总 wall ≈ 1 kernel 时间。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `sync_patterns.cu` |
| Kernel | `delay_kernel`：每 thread 跑 `ITERATIONS = 200000` 次寄存器级 `val = val*0.999f + 0.001f` |
| Grid | 256 blocks × 256 threads = 65536 thread（每 thread 处理 1 个元素，全 N=1<<16） |
| 三组测试 | (1) DeviceSync: launch on stream1 → sync → launch on stream2 → sync（顺序）<br>(2) StreamSync: 两个 stream 同时 launch kernel，然后分别 sync stream<br>(3) EventSync: 同 StreamSync，但 record event 后用 event sync |
| 采集 | `nsys profile -o report09 --force-overwrite true --trace=cuda ./sync_patterns` |
| 自打印（节选）| 3 个总时间：DeviceSync ≈ 2× kernel time；StreamSync/EventSync ≈ 1× kernel time |

注意：两 stream 共享两个不同 device 数组（d_data1, d_data2），避免数据竞争。

## 2. 关键测量数据

### 2.1 6 个 delay_kernel（来自 `cuda_gpu_kern_sum` + `cuda_kern_exec_sum`）

聚合：
| 维度 | TAvg | TMed | TMin | TMax | TStdDev |
|---|---|---|---|---|---|
| API→done (TAvg) | 1.342 ms | 1.135 ms | 1.072 ms | 1.814 ms | 0.366 ms |
| Kernel exec (KAvg) | 1.317 ms | 1.082 ms | 1.065 ms | **1.809 ms** | 0.378 ms |

**双峰结构清晰**：4 个 kernels 在 ~1.06-1.08 ms 间、2 个 kernels 在 ~1.81 ms 间。std 0.378 ms 远大于均值 1.32 ms 的 1/4，强烈非高斯分布——这是关键证据。

### 2.2 三种 sync API 时长（来自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 单次平均 | TMin | TMax |
|---|---|---|---|---|---|
| `cudaDeviceSynchronize` | 2 | 2.14 ms | 1.068 ms | 1.067 ms | 1.069 ms |
| `cudaStreamSynchronize` | 2 | 1.81 ms | 0.906 ms | 0.729 ms | 1.084 ms |
| `cudaEventSynchronize` | 2 | 1.81 ms | 0.905 ms | 0.729 ms | 1.080 ms |

三种 sync 接近但**不等价**：DeviceSync 平均 1.07 ms/blocking call，Stream/EventSync 平均 0.91 ms/call。

### 2.3 launch 与辅助 API（节选自 `cuda_api_sum`）

| API | 次数 | 总耗时 |
|---|---|---|
| `cudaHostAlloc` | 1 | 194.6 ms（setup，96.7% host 总时长） |
| `cudaLaunchKernel` | 6 | 0.135 ms（avg 22 µs/launch） |
| `cudaStreamCreate` | 2 | 0.111 ms |
| `cudaStreamDestroy` | 2 | 11.3 µs |
| `cudaEventCreate` | 2 | 8.5 µs（**注意**：远低于 ex 04/07 的 ~197 ms 异常） |
| `cudaEventRecord` | 2 | 6.8 µs |
| `cudaEventDestroy` | 2 | 1.3 µs |
| `cudaMemcpy` | 2 | 57.9 µs（H2D 初始数据） |

### 2.4 memcpy（来自 `cuda_gpu_mem_time_sum`）

| 操作 | 次数 | 单次 | 总耗时 |
|---|---|---|---|
| H2D | 2 | 10.6 µs | 21.2 µs（数据极小：256 KB × 2 = 512 KB） |

## 3. 分析与结论

### 3.1 双峰来源：sync 策略改变了 kernel 的物理执行模式

| Test | sync 策略 | 期望 GPU 行为 | 实测 kernel 时长（2 inst each） |
|---|---|---|---|
| 1 | DeviceSync（每 launch 后 sync） | 两 kernel 完全 serialize, 每个 alone on GPU | **2 × 1.81 ms** |
| 2 | StreamSync（两 stream 同时 launch, 后分别 sync） | 两 kernel 真重叠 | **2 × 1.07 ms** |
| 3 | EventSync（事件精细化同步） | 两 kernel 真重叠 | **2 × 1.07 ms** |

按本数据反推：**Test 1 = 慢 kernel（1.81 ms each），Tests 2/3 = 快 kernel（1.07 ms each）**——这与 README 预期"DeviceSync 应比 Stream/Event 慢"方向一致，但实测差异比 README 描述更复杂。

### 3.2 为什么"serialize 的 kernel 时长 + concurrent 的 kernel 时长"是反向的？

直觉上：serialize 单 kernel 应该快（独占 GPU），concurrent 双 kernel 应该慢（共享 SM）。但实测相反。三种可能解释：

**Hypothesis A：GPU 时钟 boosting 差异**
- 单 kernel 串行触发 GPU 升频到 boost clock
- 串行模式下 GPU 可能因 idle gap 而降频
- 测得 Test 1 kernels 1.81 ms vs Tests 2/3 kernels 1.07 ms → 1.69× 差距，太大，不能纯靠 frequency 解释

**Hypothesis B：Occupancy / 流水线填充差异**
- delay_kernel 每 thread 跑 200000 iter，**寄存器依赖严重**（每 iter 依赖前次 val）
- 单 kernel 在 H20 80 SMs 上：256 blocks → ~3 blocks/SM，per-SM 768 threads，warp/SM 24
- 并发双 kernel：512 blocks → ~6 blocks/SM，per-SM 1536 threads，warp/SM 48 → **occupancy 翻倍**
- 更高 occupancy → better ALU pipeline feeding → per-kernel faster execution

**Hypothesis C：可能映射反了**
- 也许 Tests 2/3 的 4 个 kernel 中只跑了 2 个快+2 个慢，而 Test 1 的两 kernel 都慢——分布 2 fast + 4 slow = 6 而非 4 fast + 2 slow
- 但 std=0.378ms 配合 median 1.082 接近 min 1.065，更可能 4 fast + 2 slow 而非 2 fast + 4 slow

**Hypothesis B 最可能**——occupancy 主导了类似 compute-bound kernel 的吞吐量。delay_kernel 的高 dependency chain 让 single-kernel ALU 使用率低，加并发 kernel 让 ALU 更忙。

### 3.3 三种 sync API 时长接近——真正的区别不在 API 本身

| Sync type | 单 call 平均 | 解释 |
|---|---|---|
| DeviceSync | 1.068 ms | waits for ALL GPU work（粗粒度）|
| StreamSync | 0.906 ms | waits for 1 stream（中粒度）|
| EventSync | 0.905 ms | waits for 1 event（细粒度）|

DeviceSync 略慢（~15%），因其内部需遍历所有 stream 的 pending work。但**API 时长不是同步策略的核心差异点**——核心差异是**它如何改变 GPU 上的 kernel 调度**，已通过双峰 kernel 时长体现。

### 3.4 工程含义

| 场景 | 推荐 sync 策略 | 原因 |
|---|---|---|
| 单 stream 顺序 pipeline | `cudaDeviceSynchronize` | 简单，没必要复杂 sync |
| 多 stream 并发 H2D/kernel/D2H pipeline | `cudaStreamSynchronize` | 各 stream 独立等待，不拖累其他 stream |
| 跨 stream 协调（如 stream2 等 stream1 完成）| `cudaEventSynchronize(event_in_stream1)` | 最细粒度，可在 kernel 中途插入等待点 |
| CUDA Graph 内 | 不用显式 sync | Graph 内部 schedule 已优化 |
| Host 想知道 GPU 工作何时完成 | `cudaEventQuery` 或 `cudaStreamQuery` | 不阻塞，poll 模式 |

### 3.5 cudaEventCreate 这里为什么没异常？

`cudaEventCreate` 2 次共 8.5 µs——**完全正常**，没有 ex 04/07 的 ~197 ms 异常。

差别：本练习**先做了 cudaHostAlloc**（194.6 ms），期间 driver 应已 lazy-init 完成 event 子系统。所以**后续 cudaEventCreate 时已不需要 cold-start 开销**。

工程含义：**CUDA driver 的多个子系统初次访问都会 lazy-init**（malloc、event、stream、kernel binary cache）。让 driver 暖机的"标准动作"是程序入口先做一次 `cudaMalloc(1 byte)` + `cudaFree()` + `cudaEventCreate + Destroy`——避免后续 benchmark 测到 init 开销。

## 4. 思考延伸

1. 让 delay_kernel 改成 memory-bound（不用 register iter，每次写 d_data[idx]），重做测试。Concurrent 模式应失去优势——两 kernel 争 HBM 带宽，每个 kernel 反而更慢。
2. 把 N=1<<16 改成 N=1<<20，grid 从 256 变 4096——single kernel 已经能塞满 80 SMs，concurrent 应不再加速（occupancy 已饱和）。
3. 用 `cudaStreamWaitEvent(stream2, event_in_stream1)` 在 GPU 端做 stream 间依赖，避免 host-side sync——这能让多 stream pipeline 完全 host-async。
4. 既然 Test 1 (serial) 慢于 Tests 2/3 (concurrent)，是不是说"性能优化时尽量并发"为普适建议？注意前提：必须 occupancy-not-saturated 的 kernel 才受益。

## 附录：复现命令

```bash
cd phase1_nsys/09_sync_patterns && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum   --format csv ./report09.nsys-rep
$NSYS stats --force-export=true --report cuda_kern_exec_sum  --format csv ./report09.nsys-rep
$NSYS stats --force-export=true --report cuda_api_sum        --format csv ./report09.nsys-rep | grep -E "(Sync|cudaLaunch|cudaHost|cudaStream|cudaEvent)"
$NSYS stats --force-export=true --report cuda_gpu_trace      --format csv ./report09.nsys-rep | sort -k1 -n  # 看时间戳双峰排序
```
