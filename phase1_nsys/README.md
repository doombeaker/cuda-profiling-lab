# Phase 1 — Nsight Systems 分析索引

本目录收录 phase1 全部 12 个 nsys 练习的参考分析文档,按练习编号顺序排列。每个子目录下的 `analysis.md` 包含:一句话结论、实验设置、关键测量数据、分析与结论、思考延伸、复现命令。

所有结论基于 **H20 (sm_90, 96 GB) + CUDA 12.4 + nsys 2023.4.4** 一手实测数据,不是推测。

---

## 索引总表

| # | 练习 | 主题 | 一句话结论 | 关键数据 |
|---|---|---|---|---|
| 01 | [Vector Add](01_vector_add/analysis.md) | 基础 kernel | kernel 自身仅 0.40 ms,而 H2D+D2H 占 97%——典型 memory-transfer-bound | kernel 0.40 ms / 传输 14.6 ms |
| 02 | [Matmul Naive vs Tiled](02_matmul/analysis.md) | 访存优化 | tiled 仅比 naive 快 1.52×,**远低于 README "several times"**——N=1024 太小 | tiled 451.8 µs / naive 686.6 µs |
| 03 | [Multi-Stream Concurrency](03_multi_stream/analysis.md) | 并发 | 4 stream wall 6.95 ms vs 串行 12.55 ms,省 44%——但 kernel 短未互相重叠 | wall 6.95 ms / serial 12.55 ms |
| 04 | [Memory Transfer Bandwidth](04_mem_transfer/analysis.md) | PCIe 带宽 | 带宽 36.7→52.6 GB/s 随 size 饱和到 PCIe Gen5 x16 理论的 82% | 4 MB→1 GB 曲线 |
| 05 | [NVTX Timeline Annotation](05_nvtx/analysis.md) | 工具教学 | `nvtx_pushpop_sum` 直接给三段时长,与 CUDA 物理操作零误差对应 | Init 877/Compute 10.9/Verify 48 ms |
| 06 | [Kernel Launch Overhead](06_kernel_launch_overhead/analysis.md) | 并发 | 单次 launch API 31.7 µs vs kernel 执行 1.13 µs——overhead 是 kernel 的 28× | noop launch 31.7 µs |
| 07 | [Pinned vs Pageable Memory](07_pinned_vs_pageable/analysis.md) | 访存优化 | pinned 比 pageable 快 5.5×,正好落 README 2-10× 区间中段 | pinned 4.87 ms / pageable 26.89 ms |
| 08 | [cuBLAS SGEMM Profiling](08_cublas/analysis.md) | 库对标 | cuBLAS 93 µs 比 tiled 451.8 µs 快 4.85×,比 naive 快 7.38× | cuBLAS 93 µs |
| 09 | [Synchronization Patterns](09_sync_patterns/analysis.md) | 同步 | 6 个 delay_kernel 呈双峰:4× 1.07 ms (并发)+ 2× 1.81 ms (串行) | 双峰 1.07/1.81 ms |
| 10 | [Unified Memory Page Faults](10_unified_memory/analysis.md) | UVM | 冷访问 12.55 ms vs 热访问 89.7 µs——140× 差距体现 page-fault 成本 | cold 12.55 ms / hot 89.7 µs |
| 11 | [nsys stats — CLI Analysis](11_nsys_stats/analysis.md) | 工具教学 | 全 12 练习 AI 友好入口;`--stats=true` 会留 stale sqlite,必须 `--force-export=true` | 三件套 covering 80% |
| 12 | [Parallel Reduction](12_reduction/analysis.md) | warp divergence | sequential 比 interleaved 快 2.37×,**远超 README 10-30%** — huge grid 放大 divergence | seq 136.3 µs / inter 322.7 µs |

---

## 横向关联

| 主题 | 相关练习 | 关联点 |
|---|---|---|
| Memory-bound 系列 | 01, 04, 07 | 三种 memory-bound 证据:kernel << transfer (01)、带宽饱和曲线 (04)、pinned 利用率 (07) |
| cuBLAS vs 手写 | **02 ↔ 08** | 同样 N=1024 SGEMM,手写 tiled 451.8 µs vs cuBLAS 93 µs——Tensor Core + 寄存器分块的鸿沟 |
| 并发 vs 同步 | 03, 09 | multi_stream (03) 重叠 memcpy 但 kernel 不重叠 ↔ sync_patterns (09) 双峰显示 kernel 并行收益 |
| UVM 冷热对比 | 10 | 单一练习内 cold/hot/prefetch 三态,可与 01、04 的显式传输成本对比 |
| 工具方法学 | **11 ↔ 全部** | ex11 是全 12 练习的 sqlite+CSV 入口教程,任何横向对比都经它统一 |

---

## AI 友好分析路径(工具方法学)

**首选组合**:`nsys export --type=sqlite` + Python `sqlite3` 直查;或者 `nsys stats --report <name> --format csv` 拿聚合表。详见 [Exercise 11](11_nsys_stats/analysis.md)。

三个高频 report 覆盖 80% 场景:
- `cuda_gpu_kern_sum` — kernel 总耗时排序
- `cuda_gpu_mem_size_sum` + `cuda_gpu_mem_time_sum` — 传输大小与带宽
- `cuda_api_sum` — CPU 侧 CUDA API 时长

**陷阱提醒**:profile.sh 用 `--stats=true` 会让后续 stats 查询返回空,必须加 `--force-export=true` 重新生成 sqlite。详见 ex11。

---

## 发现的 Lab Bug

分析过程中发现 3 个 lab 本身的问题,已记入对应 MD 的"思考延伸"段,供作者参考:

1. **ex10** `--trace=unified-memory` 在 nsys 2023.4 上为无效 flag,已最小补丁为 `--trace=cuda,osrt`;另 H20 上 `um_sum`/`um_cpu_page_faults_sum` 全部 SKIPPED,改用 kernel 时长双峰推导结论。
2. **ex11** `analyze.sh` 未提 `--force-export=true`,README 称 "Add --format csv" 但缺关键 caveat。
3. **ex02** README 称 tiled "several times faster",但 N=1024 下实测仅 1.52×——加速倍数规模依赖,需 N≥4096 才能看到数倍。

---

## 阅读建议

- **初学者**:按编号顺序 01→12,ex01 是模板,ex11 是工具入口。
- **找特定性能问题**:先看上表"主题"列定位,再点链接深入。
- **做横向对比**:用"横向关联"表,尤其 02↔08 的 cuBLAS 鸿沟和 11↔all 的工具统一性。
- **复现某练习**:进入对应子目录,运行 `bash profile.sh`,然后按该 MD 的"附录:复现命令"段执行 nsys stats 查询。
