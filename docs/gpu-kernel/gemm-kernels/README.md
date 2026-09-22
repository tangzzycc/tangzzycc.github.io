# GEMM Kernel 源码

本目录保存《GEMM CUDA Core 优化：从 Naive 到接近 cuBLAS》使用的完整 CUDA Kernel。

实现以 Simon Boehm 的 [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) 为基础，保持原始优化顺序与核心代码结构，并将说明性注释翻译为中文。原项目采用 MIT License，许可证见 [LICENSE](LICENSE)。

| Kernel | 主要变化 | 源码 |
| --- | --- | --- |
| 1 | Naive SGEMM | [1_naive.cuh](1_naive.cuh) |
| 2 | Global Memory 合并访存 | [2_kernel_global_mem_coalesce.cuh](2_kernel_global_mem_coalesce.cuh) |
| 3 | Shared Memory Cache Blocking | [3_kernel_shared_mem_blocking.cuh](3_kernel_shared_mem_blocking.cuh) |
| 4 | 一维 Thread Tiling | [4_kernel_1D_blocktiling.cuh](4_kernel_1D_blocktiling.cuh) |
| 5 | 二维 Thread Tiling | [5_kernel_2D_blocktiling.cuh](5_kernel_2D_blocktiling.cuh) |
| 6 | Shared/Global Memory 向量化 | [6_kernel_vectorize.cuh](6_kernel_vectorize.cuh) |
| 7 | 线性化布局减少 Bank Conflict | [7_kernel_resolve_bank_conflicts.cuh](7_kernel_resolve_bank_conflicts.cuh) |
| 8 | Padding 改变 Bank 映射 | [8_kernel_bank_extra_col.cuh](8_kernel_bank_extra_col.cuh) |
| 9 | Autotuning 参数版本 | [9_kernel_autotuned.cuh](9_kernel_autotuned.cuh) |
| 10 | Warp Tiling | [10_kernel_warptiling.cuh](10_kernel_warptiling.cuh) |
| 11 | 手工双缓冲 | [11_kernel_double_buffering.cuh](11_kernel_double_buffering.cuh) |
| 12 | `cuda::memcpy_async` 与 Barrier | [12_kernel_double_buffering.cuh](12_kernel_double_buffering.cuh) |

这些文件是教学用实现，主要用于观察数据布局和优化步骤。Kernel 3 之后通常假设矩阵尺寸满足 tile 的整除与对齐要求，不应直接当作支持任意形状的生产级 GEMM。
