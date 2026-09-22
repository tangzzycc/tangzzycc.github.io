# GPU Kernel

这里记录 GPU Kernel 的实现与性能优化笔记，关注线程工作划分、内存层级、数据复用和硬件调度之间的关系。

## GEMM

- [GEMM CUDA Core 优化：从 Naive 到接近 cuBLAS](GEMM-CUDA-Core优化.md)

这篇笔记主要基于 Simon Boehm 的英文博客 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) 及其 [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) 配套实现。

笔记沿用原博客的 Kernel 优化顺序和 A6000 实验主线，并参考 Hamza Elshafie 的 [H100 GEMM 优化工作日志](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) 补充 CUDA Core 通用机制与示意图。Tensor Core 路径不在本文范围内。

在中文翻译的基础上，正文补充 Roofline 推导、线程与内存映射、图片解读、尾块处理以及双缓冲分析。完整 CUDA 实现集中在 [GEMM Kernel 源码目录](gemm-kernels/README.md)。
