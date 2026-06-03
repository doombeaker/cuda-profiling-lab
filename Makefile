CUDA_HOME ?= /usr/local/cuda-12.4
NVCC       = $(CUDA_HOME)/bin/nvcc
NVCCFLAGS  = -std=c++17 -arch=sm_90 -I$(CURDIR)/common -O2

EXERCISES = \
	phase1_nsys/01_vector_add/vector_add \
	phase1_nsys/02_matmul/matmul \
	phase1_nsys/03_multi_stream/multi_stream \
	phase1_nsys/04_mem_transfer/mem_transfer \
	phase1_nsys/05_nvtx/nvtx \
	phase1_nsys/06_kernel_launch_overhead/kernel_launch_overhead \
	phase1_nsys/07_pinned_vs_pageable/pinned_vs_pageable \
	phase1_nsys/08_cublas/cublas \
	phase1_nsys/09_sync_patterns/sync_patterns \
	phase1_nsys/10_unified_memory/unified_memory \
	phase1_nsys/11_nsys_stats/nsys_stats \
	phase1_nsys/12_reduction/reduction \
	phase2_ncu/05_occupancy/occupancy \
	phase2_ncu/06_mem_bandwidth/mem_bandwidth \
	phase2_ncu/07_compute_mem_bound/compute_mem_bound \
	phase2_ncu/08_warp_divergence/warp_divergence \
	phase2_ncu/09_bank_conflicts/bank_conflicts \
	phase2_ncu/10_launch_config/launch_config \
	phase2_ncu/11_memory_hierarchy/memory_hierarchy \
	phase2_ncu/12_stall_reasons/stall_reasons \
	phase2_ncu/13_instruction_mix/instruction_mix \
	phase2_ncu/14_ncu_cli/ncu_cli \
	phase2_ncu/15_ncu_sections/ncu_sections

.PHONY: all clean

all: $(EXERCISES)

phase1_nsys/01_vector_add/vector_add:  phase1_nsys/01_vector_add/vector_add.cu  common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/02_matmul/matmul:          phase1_nsys/02_matmul/matmul.cu          common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/03_multi_stream/multi_stream: phase1_nsys/03_multi_stream/multi_stream.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/04_mem_transfer/mem_transfer: phase1_nsys/04_mem_transfer/mem_transfer.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/05_occupancy/occupancy:     phase2_ncu/05_occupancy/occupancy.cu     common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/06_mem_bandwidth/mem_bandwidth: phase2_ncu/06_mem_bandwidth/mem_bandwidth.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/05_nvtx/nvtx:                    phase1_nsys/05_nvtx/nvtx.cu                    common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/06_kernel_launch_overhead/kernel_launch_overhead: phase1_nsys/06_kernel_launch_overhead/kernel_launch_overhead.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/07_pinned_vs_pageable/pinned_vs_pageable: phase1_nsys/07_pinned_vs_pageable/pinned_vs_pageable.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/08_cublas/cublas:                 phase1_nsys/08_cublas/cublas.cu                 common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@ -lcublas

phase1_nsys/09_sync_patterns/sync_patterns:   phase1_nsys/09_sync_patterns/sync_patterns.cu   common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/10_unified_memory/unified_memory: phase1_nsys/10_unified_memory/unified_memory.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/11_nsys_stats/nsys_stats:         phase1_nsys/11_nsys_stats/nsys_stats.cu         common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase1_nsys/12_reduction/reduction:           phase1_nsys/12_reduction/reduction.cu           common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/08_warp_divergence/warp_divergence:   phase2_ncu/08_warp_divergence/warp_divergence.cu   common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/09_bank_conflicts/bank_conflicts:   phase2_ncu/09_bank_conflicts/bank_conflicts.cu   common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/10_launch_config/launch_config:     phase2_ncu/10_launch_config/launch_config.cu     common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/11_memory_hierarchy/memory_hierarchy: phase2_ncu/11_memory_hierarchy/memory_hierarchy.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/12_stall_reasons/stall_reasons:     phase2_ncu/12_stall_reasons/stall_reasons.cu     common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/13_instruction_mix/instruction_mix: phase2_ncu/13_instruction_mix/instruction_mix.cu common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/14_ncu_cli/ncu_cli:                 phase2_ncu/14_ncu_cli/ncu_cli.cu                 common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

phase2_ncu/15_ncu_sections/ncu_sections:       phase2_ncu/15_ncu_sections/ncu_sections.cu       common/error_check.h common/timer.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(EXERCISES)