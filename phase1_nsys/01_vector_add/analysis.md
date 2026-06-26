# 参考分析过程与结论 — Exercise 01: Vector Add

> 本文档演示如何**不经 nsys GUI**，仅用 AI 友好的文本格式（SQLite + CSV）从 `report01.nsys-rep` 中完整重建三段式 timeline 分析。这是 phase1 全部 12 个练习采用的统一方法论。

## 0. 一句话结论

vector_add kernel 自身仅耗 **0.40 ms**，而 H2D/D2H 内存搬运花了 **~14.6 ms**（占 GPU 总操作时长的 97%）——这是个典型 **memory-transfer-bound** 程序，优化 kernel 毫无意义，瓶颈在 PCIe。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `vector_add.cu` |
| 数据规模 | `N = 1 << 26` = 67,108,864（64M 元素） |
| 每缓冲区 | 256 MB |
| 缓冲区数量 | 3（h_a, h_b, h_c；都要走 PCIe） |
| Host 内存类型 | **pinned**（`cudaMallocHost`） |
| Kernel 配置 | `grid=262144, block=256` |
| 采集命令 | `nsys profile -o report01 --force-overwrite true ./vector_add`（默认 `--trace=cuda`） |
| 报告大小 | 734,557 字节 |
| 程序自打印 | `Vector Add (N=67108864): GPU time = 15.234 ms`，`Verification: PASS` |

## 2. AI 友好分析路径

`.nsys-rep` 是 NVIDIA 私有二进制，AI 无法直接 read。两条互补路径：

### 路径 A：导出标准 SQLite
```bash
nsys export --type=sqlite --force=true -o report01.sqlite report01.nsys-rep
# 然后可用 python3 -c "import sqlite3; ..." 任意 SQL 查询时序表
```

### 路径 B：nsys stats 直接拿预聚合 CSV report
```bash
nsys stats --report cuda_gpu_kern_sum    --format csv report01.nsys-rep  # kernel 汇总
nsys stats --report cuda_gpu_mem_time_sum --format csv report01.nsys-rep # memcpy 时长
nsys stats --report cuda_api_sum         --format csv report01.nsys-rep  # host API 汇总
```

> 路径 B 第一次调用时会自动在当前目录生成 `report01.sqlite` 副产物；后续调用复用。SQLite 给底层时序数据（含 startNs/endNs），stats CSV 给 AI-ready 聚合视图——两者互补，覆盖所有分析需求。

## 3. 关键测量数据（实测）

### 3.1 GPU 端 — Kernel + MemOps（来自 `cuda_gpu_kern_sum` + `cuda_gpu_mem_time_sum`）

| 类别 | 名称 | 次数 | 单次平均 | 总耗时 | 占 GPU 操作时长 |
|---|---|---|---|---|---|
| Kernel | `vector_add(float*, float*, float*, int)` | 1 | 0.396 ms | **0.396 ms** | 2.6% |
| Memcpy | H2D（`cudaMemcpyHostToDevice`） | 2 | 4.867 ms | **9.734 ms** | 64.0% |
| Memcpy | D2H（`cudaMemcpyDeviceToHost`） | 1 | 4.893 ms | **4.893 ms** | 32.1% |
| — | **合计（GPU operation time）** | 4 | — | **15.023 ms** | 100% |

> 加总 0.396 + 9.734 + 4.893 = 15.023 ms，与程序 GpuTimer 打印的 `15.234 ms` 仅差 0.2 ms（差异来自 launch/sync 抖动）。数据自洽。

### 3.2 Host 端 — CUDA API（来自 `cuda_api_sum`）

| API | 调用次数 | 总耗时 | 单次平均 | 备注 |
|---|---|---|---|---|
| `cudaHostAlloc` | 3 | 453.2 ms | 150.7 ms | pinned 内存一次性分配，**setup** |
| `cudaFreeHost` | 3 | 82.4 ms | 27.5 ms | pinned 内存释放，**teardown** |
| `cudaMemcpy` | 3 | 15.1 ms | 5.0 ms | 同步等待 GPU 完成 ≈ GPU 实际耗时 |
| `cudaMalloc` | 3 | ~0.4 ms | <0.2 ms | device 内存分配 |
| `cudaLaunchKernel` | 1 | <0.1 ms | — | 异步派发，几乎瞬时返回 |

## 4. 分析与结论

### 4.1 Timeline 三段式重构（不需 GUI）

用 SQLite 中 `CUPTI_ACTIVITY_KIND_MEMCPY` 与 `CUPTI_ACTIVITY_KIND_KERNEL` 表的 `startNs/endNs` 字段，按时间戳排序即得 timeline。nsys-ui 渲染的色块本质上就是这条时序：

```
[t=0]              ── H2D (h_a → d_a)  4.867 ms
[t=4.867ms]        ── H2D (h_b → d_b)  4.867 ms
[t=9.734ms]        ── vector_add kernel 0.396 ms
[t=10.130ms]       ── D2H (d_c → h_c)  4.893 ms
[t=15.023ms]       ── 完成
```

**三段式（H2D → Kernel → D2H）在数据中天然成立**——三个 cudaMemcpy 串行同步执行（因为没用 stream/async），kernel 必须等 H2D 完成才能跑，D2H 又必须等 kernel 完成才能开始。

### 4.2 Kernel 是 memory-bound（符合教学预期）

vector_add 是教科书级 memory-bound kernel：

- 算术强度 ≈ 1 FLOP / 12 bytes（读 2 个 float + 写 1 个 float）
- H20 HBM3 带宽峰值 ~4 TB/s，理论极限 ~333 GFLOPS
- 实测 kernel 0.396 ms 处理 256MB×3 = 768MB → 有效带宽 **1.94 TB/s** ≈ 峰值 48%
- 这个数字低于理论极限但合理——kernel 内部存在读写依赖、未做 prefetch，且 64M 元素让 grid 调度本身有少量尾部损耗

**关键判断**：kernel 耗时本就低于访存下限，再怎么优化 kernel 也跑不出更高吞吐——因为本质上一直是 HBM 在供电，不是 SM 算力在跑。

### 4.3 "kernel 计算只占 2.6%" 的工程含义

| 优化方向 | 理论上限收益 | 工程评价 |
|---|---|---|
| 优化 vector_add kernel | kernel 0.40ms → 0ms，总时长 15.0 → 14.6ms（**-2.6%**） | 投入产出比极差，**不要做** |
| 改用多 stream + async memcpy，使 H2D 与 kernel overlap | 三段串行 → 部分 overlap，可能压缩到 ~9.7ms（**-35%**） | **收益最大** |
| 改用 CUDA Graph 重放 | 消除 cudaMemcpy 的 host API 同步抖动；但 transfer GPU-side 时长不变 | 边际收益小 |
| 改 pinned 已用，无可榨取空间 | — | — |

**这就是 nsys 系统级视角存在的理由：先确定瓶颈层级，再决定优化深度。** 盲目优化 kernel 等于浪费时间——本例正确的下一步是去看 exercise 03（multi-stream）和 exercise 11（CI 自动化统计）。

### 4.4 Pinned memory 的隐性 setup 成本

`cudaHostAlloc` 在本程序中累计耗时 **453.2 ms**，远超 15.0 ms 的"运行时间"。但这是**一次性 setup 成本**，不计入 GpuTimer（GpuTimer 只覆盖 H2D/Kernel/D2H）。

| 时机 | API | 耗时 | 是否纳入 GpuTimer |
|---|---|---|---|
| Setup | cudaHostAlloc ×3 + cudaMalloc ×3 + host init loop | ~453ms | ❌ 排除 |
| 运行 | 2× H2D + Kernel + 1× D2H | ~15ms | ✅ 包含 |
| Teardown | cudaFreeHost ×3 + cudaFree ×3 | ~83ms | ❌ 排除 |

教学要点：
- pinned memory 的优势是 transfer 速度（避开 staging copy 见 exercise 07），代价是分配/释放开销大、锁住物理内存
- 对**短生命周期 benchmark**，pinned 分配开销会污染总时长统计——这也是为什么 GpuTimer 故意排除 setup/teardown
- 对**长生命周期长期复用**的应用（如训练循环），pinned 分配一次摊销到无数次 iteration，开销可忽略

### 4.5 PCIe 带宽利用率（顺便核对）

- 256MB × 3 / 14.627ms = **52.6 GB/s** 有效带宽
- H20 服务器 PCIe Gen5 x16 理论 ~64 GB/s（去除协议开销 ~50 GB/s 可达）
- 实测 52.6 GB/s ≈ 理论 82%、可达 ~105%——**健康的 pinned transfer**

## 5. 思考延伸（供学习者验证）

1. 把 `cudaMallocHost` 换成 `malloc`，单次 H2D 会变吗？（→ 见 exercise 07 实测：约 2× 慢，因 staging copy）
2. 改用 4 stream + `cudaMemcpyAsync`，能否真把 15ms 压到 ~9.7ms？（→ 见 exercise 03 实测，但 H2D 间存在被忽略的依赖关系：vector_add 需要两段 input 都到齐）
3. 既然 setup 占 453ms，单纯用 wall-clock 评价 "GPU efficiency" 是否误导？（→ setup/teardown 应单独列报，不要混入 GPU 工作时间）
4. vector_add 这种 2.6% kernel 的程序，加 ncu profile 有意义吗？（→ 没意义，先 nsys 找层次，ncu 只在 kernel 时间显著时才值得跑）

---

## 附录：复现命令清单

```bash
cd phase1_nsys/01_vector_add
./profile.sh                                # 生成 report01.nsys-rep

NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys
$NSYS export --type=sqlite --force=true -o report01.sqlite report01.nsys-rep
$NSYS stats --report cuda_gpu_kern_sum    --format csv report01.nsys-rep
$NSYS stats --report cuda_gpu_mem_time_sum --format csv report01.nsys-rep
$NSYS stats --report cuda_api_sum         --format csv report01.nsys-rep
```

如需重做 timeline 重建（路径 A 的 SQL）：

```python
import sqlite3
con = sqlite3.connect('report01.sqlite')
# 提取所有 GPU activity（kernel + memcpy）按 startNs 排序
for row in con.execute("""
  SELECT startNs, endNs, (endNs-startNs)/1e6 AS dur_ms, name, type
  FROM CUPTI_ACTIVITY_KIND_KERNEL
  UNION ALL
  SELECT startNs, endNs, (endNs-startNs)/1e6 AS dur_ms, copyKind, 'memcpy'
  FROM CUPTI_ACTIVITY_KIND_MEMCPY
  ORDER BY startNs
"""):
    print(row)
```
