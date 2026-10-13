---
title: "GPU与算子优化：面试问题与示例回答"
---

# GPU与算子优化：面试问题与示例回答

??? question "1. AMD 与 NVIDIA CUDA GPU 的内存层次有什么共同点和差异？"

    两者都可以按寄存器、共享片上存储、cache 和 global memory 来理解。CUDA 的 **shared memory 对应 AMD 的 LDS** ，都是 block/workgroup 内显式管理的数据复用空间；global memory 通常由显存承载，访问经过对应的 cache 路径。AMD 的特点是显式区分 **lane 私有的 VGPR 和 wave 共享的 SGPR** ，而 cache 拓扑、共享存储与 L1 的组织也和 NVIDIA 不完全相同。优化原则相通：**合并访存、提高片上复用、减少 bank conflict** ，并平衡寄存器/LDS 占用和并行度；具体容量和参数要按目标架构调整。

??? question "2. AMD Instinct MI355X 的关键规格及其对推理的意义是什么？"

    MI355X 是 AMD CDNA 4 数据中心加速器，单卡有 **288 GB HBM3E、8 TB/s 显存带宽** ，BF16/FP16 **稠密矩阵峰值约 2.5 PFLOPS** ，FP8 约 5 PFLOPS，支持 MXFP4/MXFP6。它有 256 个 CU、1024 个 Matrix Core，通过 Infinity Fabric 连接其他 GPU。推理中显存决定模型和缓存容量，带宽影响权重及 KV 读取，矩阵算力影响大 GEMM；MoE 跨卡执行还要考虑互联和负载均衡。**峰值规格不能直接换算成模型 token/s** 。

??? question "3. CUDA GEMM 算子的一般实现与优化思路是什么？"

    我一般按**访存、复用和流水线** 三层优化。先保证 warp 合并访存，再用 **shared memory 分块复用 A/B** ，让每个线程在寄存器里计算多个输出。之后调整 tile 和布局，用预取重叠搬运与计算；精度允许时使用 Tensor Core。最后根据实际 shape **平衡复用、寄存器和并行度** ，用 profiler 与同配置计时验证收益。

??? question "4. AMD RDNA 上 HIP WMMA GEMM 算子的实现与优化思路是什么？"

    RDNA 使用 WMMA 矩阵乘累加指令。以我的 gfx1151 实现为例，一个 **wave32 协作计算 `16×16×16`** ，输入 BF16、累加 FP32。小 M、窄 N 时，我让多个 wave 分担 K，在 LDS 合并结果；另一条 tiled 路径让 wave 分担输出列，共享 A，优化搬运和 LDS 布局。**CDNA 使用 MFMA，通常由 wave64 协作** 。两者都围绕分块复用、预取和资源平衡，但 fragment 布局、指令性能和硬件资源不同，迁移时需要**重新设计数据搬运、tile 和流水线** ，再用 profiler 验证。

??? question "5. FlyDSL 是什么？它与 CuTe 有什么共同点？"

    FlyDSL 是面向 AMD GPU 的 Python Kernel DSL，通过 MLIR 和 LLVM 生成设备代码。它与 CuTe 的共同点是用 **Layout 描述坐标、线程和数据的映射** ，再用 **Copy/MMA atom 组合搬运和矩阵计算** 。这样可以用更高层的表达组织 tile，同时控制寄存器、LDS 和硬件指令。区别主要在**目标硬件与后端** ：FlyDSL 面向 AMD 的 WMMA/MFMA，CuTe 属于 NVIDIA CUTLASS 生态；其中 CuTe DSL 是 Python 接口，CuTe 本身也有 C++ 模板实现。
