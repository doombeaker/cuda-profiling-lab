# Exercise 04: Memory Transfer Bandwidth

## Goal

Quantify H2D (Host-to-Device) and D2H (Device-to-Host) transfer bandwidth and determine whether a workload is memory-transfer-bound.

## Key Concepts

### Bandwidth Formula

```
bandwidth = data_size / time
```

Where `data_size` is in bytes and `time` is in seconds. For GB/s:

```
bandwidth_GBps = (bytes / (1024^3)) / (time_ms / 1000)
```

### What You'll See on the nsys Timeline

- **H2D transfers** appear as blue memory copy bars
- **D2H transfers** appear as blue memory copy bars in the reverse direction
- **Kernel execution** appears as green bars on the GPU row
- The relative widths of these bars reveal the bottleneck

### Diagnosing Memory-Transfer-Bound Workloads

A workload is **memory-transfer-bound** when the time spent moving data across the PCIe bus exceeds the time the GPU spends computing. Compare:

- `H2D time + D2H time` vs `kernel time`
- If transfer time dominates, the GPU sits idle waiting for data
- If kernel time dominates, the PCIe link has headroom

### When to Invest in Pinned Memory + Async Transfers

- **Pinned memory** (`cudaMallocHost`) already used here — it avoids a staging copy and enables full PCIe bandwidth
- **Asynchronous transfers** (`cudaMemcpyAsync`) with CUDA streams allow overlapping transfers with kernel execution
- Worth the effort when: transfer time is a significant fraction of total time AND you have independent data chunks to pipeline
- Not worth it when: kernel time already dominates and transfers are hidden by compute

## What to Look For

1. How bandwidth scales with transfer size (larger transfers approach PCIe theoretical peak)
2. Whether the trivial kernel time is negligible compared to transfer time
3. At what data size the workload shifts from compute-bound to transfer-bound
