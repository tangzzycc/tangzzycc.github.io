# GPU Kernel

这里记录 GPU Kernel 的实现与性能优化笔记，关注线程工作划分、内存层级、数据复用和硬件调度之间的关系。

## FlashRT RDNA 3.5

- [FlashRT RDNA 3.5 Kernel 源码精读](flashrt-rdna35/index.md)

这组笔记围绕 FlashRT PR #213 新增的 24 个 RDNA HIP Kernel 展开。内容从 HIP Launch、Thread/Wave/Workgroup Ownership 和 Wave32 Reduction 开始，再逐个拆解 Activation、Residual、Normalization、RoPE、Encoder/Decoder Attention 与 Small-M WMMA GEMM，并说明 Host Launcher、Python Provider 和 Fallback 共同形成的实际执行边界。

## FlyDSL

- [FlyDSL：从 Layout 到生产级 Kernel](flydsl/index.md)

这组三篇笔记先讲清 `@flyc.jit` / `@flyc.kernel` 的编译边界与 Layout/Tile 编程基础，再通过经授权迁移的《FlyDSL — 一次源码精读》，沿四个官方示例完整阅读 Vector Add、Tiled Copy、TiledMma 和 Preshuffle GEMM。

## GEMM

- [GEMM CUDA Core 优化：从 Naive 到接近 cuBLAS](GEMM-CUDA-Core优化.md)

这篇笔记主要基于 Simon Boehm 的英文博客 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) 及其 [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA) 配套实现。

笔记沿用原博客的 Kernel 优化顺序和 A6000 实验主线，并参考 Hamza Elshafie 的 [H100 GEMM 优化工作日志](https://hamzaelshafie.bearblog.dev/worklog-optimising-gemm-on-nvidia-h100-for-cublas-like-performance-wip/) 补充 CUDA Core 通用机制与示意图。Tensor Core 路径不在本文范围内。

在中文翻译的基础上，正文补充 Roofline 推导、线程与内存映射、图片解读、尾块处理以及双缓冲分析。完整 CUDA 实现集中在 GitHub 的 [GEMM Kernel 源码目录](https://github.com/tangzzycc/notebook/tree/master/docs/gpu-kernel/gemm-kernels)。
