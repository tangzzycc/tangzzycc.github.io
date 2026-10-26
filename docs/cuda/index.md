---
title: "CUDA 学习"
hide:
  - toc
  - footer
---

# CUDA 学习

依据本目录的课程讲义与配套代码，整理 CUDA 编程、GPU 访存、算子优化和性能分析的学习笔记。

## 笔记

- [Reduce 规约求和](Reduce规约求和.md)：树形归约与两阶段计算。

## 学习资料

- [课程讲义：模型部署与 CUDA 教程汇总](模型部署与cuda教程汇总.pptx)：包含 GGML 模型部署、Reduce、SGEMM、GPU 硬件与调试工具、Nsight Compute、Tensor Core、MMA、Swizzle 和 CUTLASS。
- **配套代码：universe_best_cuda_practice**：已解压至 `code/universe_best_cuda_practice/`，包含逐级优化的 CUDA kernels、CMake 工程和测试示例。[原始压缩包](universe_best_cuda_practice.zip)

## 资料主题

- **Reduce**：全局内存与共享内存归约、分支与 bank conflict、循环展开、Warp Shuffle。
- **SGEMM**：共享内存分块、线程计算量、寄存器分块、向量化访存和双缓冲。
- **性能分析**：矩阵转置、合并访存、Roofline 与 Nsight Compute 指标。
- **Tensor Core / WMMA**：矩阵 fragment、Warp 分块、双缓冲和异步搬运。
- **MMA / Swizzle**：PTX 矩阵指令、数据布局、共享内存 padding 与 swizzle。
- **CUTLASS**：GEMM 调用、分层计算结构和 Epilogue。
- **FlashAttention**：分块 Attention、Softmax 与 CUDA/PyTorch 示例。
