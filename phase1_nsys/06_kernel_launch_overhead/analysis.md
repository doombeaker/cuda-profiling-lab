# 参考分析过程与结论 — Exercise 06: Kernel Launch Overhead

> 沿用 exercise 01 的 AI 友好路径。本练习命题：**launch overhead 究竟多大？批量 launch 是否能摊薄？** 用 `cuda_kern_exec_sum` 的"API time / queue / kernel exec"三段分解，给出数字答案。

## 0. 一句话结论

单次 `noop_kernel<<<1000,1>>>()` 实测 **API 调用 31.7 µs，kernel 执行 1.13 µs —— launch overhead 是 kernel 自身的 28×**。批量 100 个 `tiny_add` 时，每 launch 的 API 时间压到 **2.7 µs（降低 12×）**——但 100 launches 合计 270 µs 仍是 100 个 kernel 合计 130 µs 的两倍，**launch overhead 仍是主导项**。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `kernel_launch_overhead.cu` |
| noop kernel | 空函数，纯测 launch 开销 |
| tiny_add kernel | 256 thread/block × 4 block = 1024 thread；每 thread 1 fp32 add |
| 数据规模 | `N_ELEMS = 1024` 仅 4 KB —— 让 kernel 极短 |
| 三组实验 | (1) `noop_kernel<<<g,1>>>` 跑 g={1,10,100,1000} 各一次（同步）；(2) 单次 `tiny_add`；(3) 100 次连续 `tiny_add` |
| 采集 | `nsys profile -o report06 --force-overwrite true ./kernel_launch_overhead` |

## 2. 关键测量数据

### 2.1 小 kernel 与 noop 的耗时分解（来自 `cuda_kern_exec_sum`）

| Kernel | 实例数 | TAvg (API→kernel done) | TMin | TMax | **AAvg** (API call) | **KAvg** (kernel exec) |
|---|---|---|---|---|---|---|
| `noop_kernel()` | 4 | 34.5 µs | 6.1 µs | 118.4 µs | **31.7 µs** | **1.13 µs** |
| `tiny_add` 单 launch | 1 | — | — | — | (single) | — |
| `tiny_add` 100 连续 | 100 | 6.47 µs | 5.78 µs | 17.2 µs | **2.71 µs** | **1.30 µs** |

`AAvg` = `cudaLaunchKernel` API 调用本身在 host 线程上的耗时
`KAvg` = kernel 在 GPU 上的实际执行耗时
**Launch overhead = AAvg**（API call 没做任何 GPU 工作，纯 dispatch）

### 2.2 host API 时长（节选自 `cuda_api_sum`）

| API | 次数 | 总耗时 | 单次平均 |
|---|---|---|---|
| `cudaMalloc` | 3 | 197.4 ms | 65.8 ms（**setup 99.7% host 总时长**） |
| `cudaLaunchKernel` | 105 | 400.1 µs | 3.81 µs（**批量被严重摊薄**） |
| `cudaFree` | 3 | 140.7 µs | 46.9 µs |
| `cudaMemcpy` | 3 | 48.8 µs | 16.3 µs |
| `cudaDeviceSynchronize` | 6 | 42.7 µs | 7.1 µs |
| `cudaEventRecord` | 12 | 26.2 µs | 2.2 µs（GpuTimer.start/stop） |
| `cudaEventSynchronize` | 6 | 10.9 µs | 1.8 µs |

## 3. 分析与结论

### 3.1 noop kernel：launch overhead 实测 28× 于 kernel

4 次 noop launch（grid=1,10,100,1000）平均情况：

| 维度 | 数值 |
|---|---|
| `cudaLaunchKernel` API 调用 | **31.7 µs** |
| queue waiting | ~0.5 µs（kernel 立即被调度）|
| kernel 真在 GPU 上跑 | **1.13 µs** |
| 总流程 | ~34.5 µs |

**API time 是 kernel 的 28×**。对于 noop kernel，"compute time" 几乎就是 "launch overhead"。

TMax=118 µs 是 grid=1000 那次（需要分发更多 block 到 SM）。TMin=6 µs 是 grid=1 那次（最快派发）—— **grid size 也影响 launch overhead**，每多一个 block 都在 driver 里加一小段工作（block descriptor 准备、SM 分配决策）。

### 3.2 tiny_add 批量 100 次：launch overhead 跌 12×

| 维度 | 单 launch | 100 launch 重 |
|---|---|---|
| AAvg (API call) | ~31.7 µs（推断同 noop） | **2.71 µs** |
| KAvg (kernel exec) | — | 1.30 µs |
| 100 launch 累计 API time | 3170 µs (假设) | **271 µs** |
| 100 kernel 累计 exec time | — | 130 µs |
| **100 launch 总耗时** | — | **≈ 400 µs** |

100 launches 实际总耗时 ≈ cudaDeviceSynchronize 之后的 total ≈ 100 × (2.71 + 1.30) ≈ 401 µs。但若每次都是 cold launch，应该是 100 × 33 = 3300 µs。**性能差距 8×**。

**为什么批量摊薄？** CUDA driver 内部维护 work submission pipeline：host 调用 `cudaLaunchKernel` 把命令写入 command buffer，driver 异步把它们推到 GPU。一旦 pipe 装满，后续 launch 命令几乎免费。**只要不 `cudaDeviceSynchronize` 中断 pipe，连续 launch 会被严重摊薄**。

### 3.3 工程含义：何时 launch overhead 真的伤你？

| 模式 | launch overhead 占比 | 评价 |
|---|---|---|
| 单次大 kernel（ex 01: vector_add 0.40 ms） | 31.7 / 400 = **8%** | 不显著 |
| 单次 noop kernel | 31.7 / 1.13 = **28×** | 灾难性，但 noop 没有实际工作 |
| 100 个微小 kernel + sync | 271 / 130 = **2.1×** | 显著浪费，应改 CUDA Graph 或合并 kernel |
| 100 个连续 launch 不 sync | 271 / 130 = 2.1× | 仍显著，**应避免在 latency-sensitive 路径上做 100+ launches** |

**优化手段**：
1. **CUDA Graph**：把 100 个 launch 录制成 graph，重放时 driver 一次提交全部，几乎消除每 launch 的 host 开销
2. **合并 kernel**：把 100 个小 kernel 合并成 1 个大 kernel（用 thread blockIdx 决定哪种工作）
3. **避免过度同步**：让 launch 自然 pipeline，sync 只在真正需要时
4. **persistent kernel**：kernel 启动后常驻，host 通过 shared memory / global flag 推送工作

### 3.4 launch overhead "5-15 µs" 的常识数字 vs 本机实测

NVIDIA 文档常说 launch overhead "typically 5-15 µs"。本机实测 H20：
| 场景 | 实测 overhead | 与文档对比 |
|---|---|---|
| cold launch (noop, grid=1-1000) | 6-118 µs（avg 31.7） | 高于文档上限 |
| pipelined launch (100 tiny_add) | 2.7 µs | 低于文档下限 |

**H20 server 上的 driver 似乎更倾向"首次 launch 昂贵、后续极便宜"的策略**。这就是为什么 warmup 对测量至关重要。

## 4. 思考延伸

1. 把 4 次 noop launch 之间不做 sync，全部跑完再 sync——平均 launch overhead 会跌到多少？（应接近 pipelined 的 2-3 µs）
2. 用 `cudaLaunchKernelExC` 配合 `cudaLaunchAttribute` 启用 cluster launch，launch overhead 会增加多少？
3. 把 4 次 noop 改成 CUDA Graph 重放，整个 graph launch overhead 应该接近 1 个 launch 的 31.7 µs（不管节点数）—— 验证一下。
4. H20 driver 版本不同会影响 launch overhead 吗？查 `nvidia-smi --query-gpu=driver_version`。

## 附录：复现命令

```bash
cd phase1_nsys/06_kernel_launch_overhead && ./profile.sh
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS stats --force-export=true --report cuda_kern_exec_sum --format csv ./report06.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_kern_sum   --format csv ./report06.nsys-rep
$NSYS stats --force-export=true --report cuda_api_sum         --format csv ./report06.nsys-rep | grep -E "(cudaLaunchKernel|cudaDeviceSynchronize)"
```

`cuda_kern_exec_sum` 是本练习的明星 report：它把单次 launch 分解成 AAvg / queue / KAvg 三段，一眼看出 overhead 占比。
