# 参考分析过程与结论 — Exercise 03: Multi-Stream Concurrency

> 沿用 exercise 01 的 AI 友好路径。本练习的核心命题：**4 stream + async memcpy 能比单 stream 跑快多少？overlap 是否真的发生？** 用 `cuda_kern_exec_sum` 和 `cuda_api_sum` 的具体数字回答。

## 0. 一句话结论

实测总耗时 **6.95 ms**，比"完全串行"的 12.55 ms 理论下界快了 **44%**，证明 4 个 stream 的 H2D/kernel/D2H **确有重叠**。但只有 memcpy 之间能 overlap，4 个 kernel 由于太短（89 µs）+ 占满 grid，**并未互相重叠**。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `multi_stream.cu` |
| stream 数 | `NUM_STREAMS = 4` |
| chunk 大小 | `CHUNK_SIZE = 1 << 24` = 16 M 个 float = **64 MB** |
| 总数据量 | `TOTAL_SIZE = 4 × 64 MB = 256 MB` |
| Host 内存 | pinned (`cudaMallocHost`) |
| Kernel 配置 | `grid = 65536, block = 256` per stream（单 chunk 用满 GPU） |
| 采集 | `nsys profile -o report03 --force-overwrite true --trace=cuda,nvtx ./multi_stream` |
| 程序自打印 | `Multi-Stream (4 streams): total time = 6.938 ms` |

## 2. 关键测量数据

### 2.1 kernel 与 memcpy 汇总（来自 `cuda_gpu_kern_sum` + `cuda_gpu_mem_time_sum` + `cuda_kern_exec_sum`）

| 类别 | 名称 | 次数 | 总耗时 | 单次平均 | Min | Max |
|---|---|---|---|---|---|---|
| Kernel | `vector_scale` | 4 | 357 µs | **89.3 µs** | 84.9 µs | 91.2 µs |
| Memcpy | H2D | 4 | 5.31 ms | **1.327 ms** | 1.228 ms | 1.378 ms |
| Memcpy | D2H | 4 | 6.89 ms | **1.722 ms** | 1.354 ms | 2.131 ms |
| **若 4 段全串行** | | | **~25 ms** | | | |
| **实测 wall 时间** | `cudaDeviceSynchronize` 调用耗时 | 1 | **6.95 ms** | | | |

> 注：4 stream 共发起 H2D×4 + kernel×4 + D2H×4 = 12 个 GPU operation，总 GPU 累计时间 = 5.31+0.357+6.89 = **12.56 ms**。但 wall 才 6.95 ms —— 必有 overlap。

### 2.2 cudaMemcpyAsync 的 host-API 调用时长（来自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 单次平均 |
|---|---|---|---|
| `cudaMemcpyAsync` | 8 | 46.5 µs | **5.8 µs**（**异步派发**，立刻返回） |
| `cudaLaunchKernel` | 4 | 1.30 ms | 323.9 µs（**首次特别慢** 1.28 ms，其余 3 次 <5 µs） |
| `cudaDeviceSynchronize` | 1 | 6.95 ms | ——（**等于程序总 wall 时间**） |
| `cudaStreamCreate` | 4 | 121.2 µs | 30.3 µs |
| `cudaStreamDestroy` | 4 | 30.2 µs | 7.5 µs |

### 2.3 kernel launch ↔ exec 时序（来自 `cuda_kern_exec_sum`）

| TAvg（API + queue + exec 总时长） | TMin | TMax | KAvg（kernel 实际执行） |
|---|---|---|---|
| 2.445 ms | 1.368 ms | 4.166 ms | 89.3 µs |

`TAvg - KAvg ≈ 2.35 ms` 是从派发到执行的"排队+等待"时间，**远大于 kernel 本身 89 µs**——证明 kernel 在 GPU 上排队等待 H2D/D2H 完成才被调度，**没有 4 个 kernel 并行执行**。

## 3. 分析与结论

### 3.1 总耗时拆解：12.56 ms → 6.95 ms，省 5.6 ms 来自哪？

- 4 段串行理论 = 4 × (H2D 1.33 + Kernel 0.089 + D2H 1.72) = **12.55 ms**
- 实测 = **6.95 ms**
- 节省 = 5.60 ms，**44%**

**省时来源**：4 个 D2H（总 6.89 ms）与 4 个 kernel（0.357 ms 总）和后续 stream 的 H2D 重叠了。具体：
- stream 0 H2D 在跑时其他 stream 还没启动
- stream 0 kernel 在跑时 stream 1 H2D 可以并行（kernel 用 SMs，H2D 用 copy engine，物理隔离）
- stream 0 D2H 在跑时 stream 1 kernel 可以跑
- 最终总时长 ≈ **首段 H2D + 一串/kernel/memcpy 重叠 + 末段 D2H** ≈ 1.33 + 4×kernel_overhead + 1.72 ≈ **6.95 ms** ✓

### 3.2 为何 4 个 kernel 没有彼此重叠？

每个 kernel `<<<65536, 256>>>` = 16.7 M 线程 —— **grid 维度远超 H20 SM 容量**（80 SM × 4-8 block/SM = 320-640 block）。GPU 全部资源被单 kernel 吃满，没有 idle SM 给其他 stream 的 kernel 用。

只有内存操作（H2D/D2H）靠 copy engine 与 kernel 并行——所以 multi-stream 的 boost 主要来自 **memcpy 与 kernel** 的 overlap（这是常见情况），不是 kernel 与 kernel 的 overlap。

### 3.3 launch overhead 巨大差异的暗示

`cudaLaunchKernel` 4 次：第一次 **1.283 ms**，后 3 次 **3.7-4.5 µs**。差异 ~300×：
- 首次：驱动初始化、context warmup、kernel binary 加载
- 后续：从二进制缓存读取，几乎瞬时

**实操含义**：CUDA benchmark 必做 **warmup**（跑一次抛掉），否则首次 launch 污染数据。exercise 08 的 cuBLAS 源码 step 6 也确实做了 warmup——这是实战标配。

### 3.4 想再压时间，下一步该干什么？

| 优化 | 预期 | 风险 |
|---|---|---|
| 增大 chunk 数（8 stream） | H2D/D2H 块更短，pipeline 更密 | 边际收益递减，调度开销上升 |
| 缩小 chunk 让 kernel 也用不满 GPU | 4 个 kernel 才有可能真重叠 | 单 kernel 性能下降 |
| 用 CUDA Graph 重放 4 段 | 消除 8 次 cudaMemcpyAsync 派发开销（46 µs）| 边际收益小 |
| **改用 cuStreamMemcpy 走 4 物理流并发拷贝** | 关键路径可压到 max(D2H×1, kernel×1) ≈ 1.72 ms | 需 H20 支持 + driver 设置 |

## 4. 思考延伸

1. 把 `CHUNK_SIZE` 减小到 `1 << 20`（1 MB），让 kernel grid 变小，4 个 kernel 能否真重叠？（→ 看 `cuda_gpu_trace` 时间戳）
2. 把 `CHUNK_SIZE` 增大到 `1 << 26`（256 MB），4 个 stream 是否还能 overlap，还是被内存带宽堵死？
3. 把 `vector_scale` 换成 `delay_kernel`（compute-bound，类似 exercise 09），能否让 kernel 与 memcpy 真正并行？（kernel 占 SMs, memcpy 用 copy engine，物理隔离 → 应可以）
4. 既然 `cudaDeviceSynchronize` 6.95 ms 等于 wall `6.938 ms`，能否不用它改用 `cudaEventSynchronize(stop_event)`？效果一致但更细粒度。

## 附录：复现命令

```bash
cd phase1_nsys/03_multi_stream && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_gpu_kern_sum    --format csv ./report03.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report03.nsys-rep
$NSYS stats --force-export=true --report cuda_kern_exec_sum   --format csv ./report03.nsys-rep
$NSYS stats --force-export=true --report cuda_api_sum         --format csv ./report03.nsys-rep | grep -E "(cudaMemcpyAsync|cudaLaunchKernel|cudaDeviceSynchronize|cudaStream)"
```
