# 参考分析过程与结论 — Exercise 11: nsys stats — CLI Analysis

> 本练习**自身就是 nsys stats 教学演示**。其他 11 个练习都用 nsys stats 做分析，本练习站在"演示者"视角，把 nsys stats 的常用 report、format、学习者陷阱、CI 集成模式梳理一遍——并**额外揭示一个 README 没提的隐藏陷阱**。

## 0. 一句话结论

`nsys stats` 是 phase1 全部 12 个练习 AI 友好分析的统一入口——三件套 `cuda_gpu_kern_sum` + `cuda_gpu_mem_size_sum` + `cuda_api_sum` 覆盖 80% 场景。但 profile.sh 用 `--stats=true` 会让后续 stats 查询返回空，**必须用 `--force-export=true` 重新生成 sqlite**，这是 README 和 analyze.sh 都没提的关键陷阱。

## 1. 实验设置

| 项目 | 值 |
|---|---|
| 源码 | `nsys_stats.cu` |
| 数据量 | `N = 1 << 24` = 16 M float = **64 MB per buffer** |
| 5 阶段 | (1) H2D 双 buffer；(2) `vector_add`；(3) `vector_scale`；(4) `vector_saxpy`；(5) D2H 主结果 |
| 采集（special） | `nsys profile -o report11 --force-overwrite true --trace=cuda **--stats=true** $EXE` |
| 额外脚本 | `analyze.sh` —— profile 后程序作者配套的 nsys stats 演示，跑 3 个 report，默认 table format（无 csv） |

`--stats=true` 是 ex 11 的特殊 flag：profile 完成后**自动调用** `nsys stats` 输出全部 report，方便用户直接在终端看。但这会**让后续手动 stats 查询返回空数据**——见 §3.1 trap。

## 2. 关键测量数据（来自 `cuda_gpu_kern_sum` + `cuda_gpu_mem_size_sum`）

### 2.1 3 个 kernel（来自 `cuda_gpu_kern_sum`）

| Kernel | 单次 | 数据量 | 算术 |
|---|---|---|---|
| `vector_saxpy` | **101.2 µs** | 64 MB 读 + 64 MB RW | `b[i] = s*a[i] + b[i]`（3 access / FMA）|
| `vector_add` | **97.3 µs** | 64 MB × 2 读 + 64 MB 写 | `c[i] = a[i] + b[i]`（3 access / 1 add）|
| `vector_scale` | **88.5 µs** | 64 MB R/W | `a[i] *= s`（2 access / 1 mul） |

每个 kernel 都跑了 1 次。时长大致与访问字节数正比，符合 memory-bound kernel 直觉。

### 2.2 memcpy（来自 `cuda_gpu_mem_size_sum` + `cuda_gpu_mem_time_sum`）

| 操作 | 次数 | 总数据量 | 单次 | 时长 | 带宽 |
|---|---|---|---|---|---|
| H2D | 2 | 134.2 MB | 64 MB | 1.218 ms × 2 = **2.44 ms** | 52.6 GB/s |
| D2H | 1 | 64 MB | 64 MB | 1.36 ms | 47.1 GB/s |

### 2.3 host API（节选自 `cuda_api_sum`）

| API | 次数 | 总耗时 |
|---|---|---|
| `cudaHostAlloc` | 3 | 269.8 ms (89.5%) — pinned setup |
| `cudaFreeHost` | 3 | 26.3 ms (8.7%) — teardown |
| `cudaMemcpy` | 3 | 4.12 ms (1.4%) — 同步等待 ≈ GPU 时间 |
| `cudaMalloc` | 3 | 0.44 ms |
| `cudaFree` | 3 | 0.46 ms |
| `cudaLaunchKernel` | 3 | 0.19 ms |

## 3. 分析与结论

### 3.1 ★ 重要学习者陷阱：`--stats=true` 与 `--force-export=true` 的交互

**症状**：profile.sh 跑完，`./report11.nsys-rep` 已生成；执行 `nsys stats --report cuda_gpu_kern_sum --format csv ./report11.nsys-rep` 返回**全空**。

**原因**：
1. `--stats=true` 让 nsys 在 profile 末尾自动生成 `report11.sqlite` 副产物
2. 后续 `nsys stats` 命令发现 `report11.sqlite` 已存在，**默认不再重生**，尝试复用
3. **但 sqlite 似乎处于未完成或正在被使用的状态**（autostats 可能未完整关闭 sqlite handle），导致 stats 报告空内容

**修复**：每次手动 stats 查询都加 `--force-export=true` 强制重新生成 .sqlite：
```bash
$NSYS stats --force-export=true --report cuda_gpu_kern_sum --format csv ./report11.nsys-rep
```

**这个陷阱对 README 是空白**——README 说 "Try the --format csv option for machine-readable output" 但没说在 ex 11 的特殊 setup 下必须加 force-export。**这是本 MD 给学习者的最重要提示。**

### 3.2 analyze.sh 的 3 个 report 与本 MD 用的对比

`analyze.sh`（478 字节）做的事：
```bash
#!/bin/bash
NSYS=$CUDA_HOME/nsight-systems-2023.4.4/bin/nsys
REPORT=./report11.nsys-rep

$NSYS stats --report cuda_gpu_kern_sum   $REPORT    # 默认 table format
$NSYS stats --report cuda_gpu_mem_size_sum $REPORT
$NSYS stats --report cuda_api_sum        $REPORT
```

| 维度 | analyze.sh | 本 MD 推荐用法 |
|---|---|---|
| `--format` | 默认（table） | `csv`（AI 友好） |
|--force-export| ❌ 没有 | ✅ 加上（应对 `--stats=true` 陷阱） |
| `--report` 种类 | 3 个（kern_sum / mem_size_sum / api_sum） | 同 3 个，但需要时再加 `cuda_gpu_mem_time_sum` / `cuda_kern_exec_sum` |
| 输出对 AI 可用性 | table 是空格对齐，AI 解析麻烦 | csv 标准，split(',') 即可 |
| 推荐 CI 用法 | 不推荐 | `--format csv` + `--force-export=true` |

**README 中"Add --format csv for machine-readable output" 只点了一半——还需 force-export。**

### 3.3 哪些 report 是 phase1 12 个练习的常驻三件套

| Report | 含义 | 用到的练习 |
|---|---|---|
| `cuda_gpu_kern_sum` | GPU kernel 汇总（每种 kernel 的总时长/次数/avg/min/max） | 全部 12 个 |
| `cuda_gpu_mem_time_sum` | GPU memcpy 时长汇总 | 凡有数据搬运的（01, 03, 04, 07, 11） |
| `cuda_api_sum` | Host 端 CUDA API 时长（含 setup/sync/launch） | 全部 12 个 |

| 报告 | 用途 | 用到的练习 |
|---|---|---|
| `cuda_gpu_mem_size_sum` | memcpy 数据量 + 带宽反算 | 04（带宽曲线）, 07（pinned vs pageable）|
| `cuda_kern_exec_sum` | launch ↔ exec 三段分解（API / queue / kernel） | 06（launch overhead）, 09（sync pattern）|
| `nvtx_pushpop_sum` | NVTX range 时长 | 05（NVTX）|
| `um_sum` 等 | UVM page fault 统计 | 10（UVM，**本机 nsys 2023.4 不可用**） |
| `cuda_gpu_trace` | per-instance 时间戳 | 03, 09, 10（需重建 timeline 时）|

### 3.4 CI 自动化模式：把 nsys stats 输出纳入 perf regression

```bash
# run.sh: CI 中每次 PR 跑一次，profile → stats → 比较基线
#!/bin/bash
set -e
NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys

# 1. profile
$NSYS profile -o ./ci_report --force-overwrite true --trace=cuda ./your_program

# 2. 抽取关键 KPI 到 csv
$NSYS stats --force-export=true --report cuda_gpu_kern_sum \
            --format csv --output ./kern_kpi.csv ./ci_report.nsys-rep

# 3. 与 baseline 比较
python3 ci_check.py kern_kpi.csv baseline.csv --threshold 0.10  # ±10% 内 OK
```

**`ci_check.py` 模式（示意）**：
```python
import csv, sys
with open(sys.argv[1]) as f:
    cur = {r['Name']: float(r['Total Time (ns)']) for r in csv.DictReader(f)}
with open(sys.argv[2]) as f:
    base = {r['Name']: float(r['Total Time (ns)']) for r in csv.DictReader(f)}

for kern, base_t in base.items():
    if kern not in cur:
        print(f"[FAIL] {kern} disappeared"); sys.exit(1)
    cur_t = cur[kern]
    delta = (cur_t - base_t) / base_t
    status = "OK  " if abs(delta) < 0.10 else "FAIL"
    print(f"[{status}] {kern}: {cur_t/1e6:.3f} ms ({delta*100:+.1f}% vs base)")
```

这样 PR 引起 >10% kernel 时长回归会 fail CI。本 lab 12 个练习的 stats CSV 都可直接套此模板。

### 3.5 程序自打印 vs nsys stats 的差距

`nsys_stats.cu` 用 GpuTimer 打印各阶段时长。两者来源对比：
| 阶段 | 程序 GpuTimer | nsys stats 实测 |
|---|---|---|
| H2D (a+b) | 自打印 | 2 个 H2D 在 mem_time_sum 各 1.22 ms |
| vector_add | 自打印 | 97.3 µs（kern_sum 直接给）|
| vector_scale | 自打印 | 88.5 µs |
| vector_saxpy | 自打印 | 101.2 µs |
| D2H (c) | 自打印 | 1.36 ms |

两者应几乎相等（都是 CUPTI 时间源）。差异 <5% 是测量噪声。

nsys stats 的**额外价值**：
- 不需要修改源码加 timer
- 一次性获取所有 kernel + 所有 memcpy + 所有 API 的全景
- 可以查 per-instance 时间戳（cuda_gpu_trace），GpuTimer 只能测一段
- 程序若用第三方库（cuBLAS / cuDNN），GpuTimer 无法注入但 nsys 能看到所有 kernel

## 4. 思考延伸

1. 编一个 CI 脚本：每次 commit 自动跑 ex 01-12 全部 profile + stats，对比 main 分支基线，发现 >10% 回归时 alarm。
2. 把 `analyze.sh` 改造：加 `--force-export=true` + `--format csv` + 重定向到文件，让它的输出 AI 友好。
3. 比较 `nsys stats --format csv` 与 `nsys stats --format column` —— 后者输出对齐但仍是文本，差异在哪里？哪个更适合 awk/grep 后处理？
4. 用 `nsys export --type=sqlite` 把 12 个 .nsys-rep 全转 sqlite 后，写 SQL `SELECT * FROM ... JOIN ...` 跨练习比较——这是比 stats 更灵活的路径。
5. `analyze.sh` 默认 table format 对教学有价值（人类阅读），csv 对 CI 有价值。能否加 `--format` 参数让 analyze.sh 两种模式都支持？这是 README 应该补的工程改进。

## 附录：复现命令

```bash
cd phase1_nsys/11_nsys_stats
./profile.sh                       # 生成 report11.nsys-rep + report11.sqlite（被 --stats=true 自动跑）

NSYS=/usr/local/cuda-12.4/nsight-systems-2023.4.4/bin/nsys

# ★ 必须加 --force-export=true 才能拿到 stat 数据，否则返回空
$NSYS stats --force-export=true --report cuda_gpu_kern_sum    --format csv ./report11.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_size_sum --format csv ./report11.nsys-rep
$NSYS stats --force-export=true --report cuda_gpu_mem_time_sum --format csv ./report11.nsys-rep
$NSYS stats --force-export=true --report cuda_api_sum         --format csv ./report11.nsys-rep

# 也可继续用作者配套的 analyze.sh，但默认 table format 不便 AI 消费
source analyze.sh
```
