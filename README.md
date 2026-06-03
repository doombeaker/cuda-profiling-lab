# nv_profiling — CUDA Profiling 系统学习项目

通过 **23 个 hands-on 练习** 系统学习 NVIDIA NSight Systems（`nsys`）和 NSight Compute（`ncu`）。

## 前置条件

- NVIDIA GPU（sm_90，如 H100/H200）
- CUDA Toolkit 12.4（含 NSight Systems 2023.4.x 和 NSight Compute 2024.1.x）
- 能用 GUI 的环境（`nsys-ui` / `ncu-ui`）查看报告

## 快速开始

```bash
# 编译全部练习
make

# 进入任一练习
cd phase1_nsys/01_vector_add
./profile.sh                       # nsys 采集
nsys-ui ./report01.nsys-rep        # GUI 查看
```

## 项目结构

```
nv_profiling/
├── Makefile            # 统一构建（sm_90, C++17, O2）
├── common/             # 公共工具（CUDA_CHECK 宏, GpuTimer, CpuTimer）
│
├── phase1_nsys/        # Phase 1：NSight Systems（系统级 profiler）
│   ├── 01_vector_add              Timeline 三段式（H2D → Kernel → D2H）
│   ├── 02_matmul                  naive vs tiled kernel 耗时对比
│   ├── 03_multi_stream            4-stream 并发重叠分析
│   ├── 04_mem_transfer            PCIe 带宽测量（5 种数据量）
│   ├── 05_nvtx                    [新增] NVTX 标注 + 彩图区间
│   ├── 06_kernel_launch_overhead  [新增] kernel launch 开销测量
│   ├── 07_pinned_vs_pageable      [新增] pinned vs pageable 带宽对比
│   ├── 08_cublas                  [新增] cuBLAS SGEMM 库级分析
│   ├── 09_sync_patterns           [新增] Device/Stream/Event 同步
│   ├── 10_unified_memory          [新增] UVM page fault / migration
│   ├── 11_nsys_stats              [新增] `nsys stats` CLI 统计
│   └── 12_reduction               [新增] interleaved vs sequential reduce
│
└── phase2_ncu/         # Phase 2：NSight Compute（kernel 级 profiler）
    ├── 05_occupancy               寄存器压力 → occupancy
    ├── 06_mem_bandwidth           coalesced vs strided 访存
    ├── 07_compute_mem_bound       Compute/Memory bound 诊断 + Roofline
    ├── 08_warp_divergence         [新增] Branch efficiency，warp divergence
    ├── 09_bank_conflicts          [新增] shared memory bank conflicts
    ├── 10_launch_config           [新增] block size 扫描调优
    ├── 11_memory_hierarchy        [新增] L1/L2/HBM 命中率分析
    ├── 12_stall_reasons           [新增] Long Scoreboard / SB / Barrier
    ├── 13_instruction_mix         [新增] FP32 / INT / Control 指令占比
    ├── 14_ncu_cli                 [新增] `ncu --csv --print-summary`
    └── 15_ncu_sections            [新增] `--section` 精确采集
```

## 学习路径

### Phase 1：NSight Systems — 系统级视角

从头到尾跑完这 12 个练习，你会：

| 能力 | 对应练习 |
|---|---|
| 读懂 Timeline（H2D/Kernel/D2H） | 01 |
| 对比不同 kernel 版本的性能 | 02, 12 |
| 分析 multi-stream 并发重叠 | 03 |
| 测量 PCIe 带宽瓶颈 | 04, 07 |
| 用 NVTX 标注应用阶段 | 05 |
| 理解 CPU-side kernel launch 开销 | 06 |
| 分析 cuBLAS 等库的 kernel 行为 | 08 |
| 比较不同同步模式的时间线差异 | 09 |
| 观察 Unified Memory page fault | 10 |
| 用 `nsys stats` 做命令行分析 | 11 |

### Phase 2：NSight Compute — Kernel 级视角

| 能力 | 对应练习 |
|---|---|
| 分析 occupancy 与寄存器压力 | 05 |
| 分析全局内存 coalescing 效率 | 06 |
| 判断 kernel 是 compute 还是 memory bound | 07 |
| 检测 warp divergence / branch efficiency | 08 |
| 检测 shared memory bank conflicts | 09 |
| 扫描最优 block size | 10 |
| 分析 L1/L2/HBM 命中率 | 11 |
| 诊断 warp stall（Long Scoreboard / SB / Barrier） | 12 |
| 分析指令类型占比（FP32/INT/Control） | 13 |
| 纯命令行 ncu 分析（`--csv`, `--print-summary`） | 14 |
| 按需选择 `--section` / `--set` 精确采集 | 15 |

## 每个练习怎么用

每个练习目录包含三个文件：

```
exercise/
├── foo.cu        # CUDA 源码
├── profile.sh    # 一键 nsys/ncu 采集脚本
└── README.md     # 学习目标、概念讲解、怎么看结果、思考题
```

三步走：

```bash
cd phase2_ncu/08_warp_divergence

# 1. 编译
make -C ../..

# 2. 采集 profiling 数据
./profile.sh

# 3. 阅读 README，按指引在 GUI 中查看
ncu-ui ./report08.ncu-rep    # ncu 报告
# 或
nsys-ui ./report08.nsys-rep  # nsys 报告
```

每个 README 会告诉你：**看哪个面板、看哪个指标、正常值是什么、异常说明什么问题**。

## 完成这个项目后

你将能够独立完成：

- 用 **nsys** 诊断 CPU-GPU 交互瓶颈、PCIe 带宽、stream 并发、UVM 迁移
- 用 **ncu** 诊断 kernel 内部瓶颈：occupancy、访存模式、bank conflicts、warp stall、指令配比
- 用 **ncu --csv / nsys stats** 做 CI 自动化 profiling