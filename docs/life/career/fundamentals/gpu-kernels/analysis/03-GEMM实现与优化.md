---
title: "GEMM 实现与优化：CUDA、RDNA 与 CDNA"
---

# GEMM 实现与优化：CUDA、RDNA 与 CDNA

## 1. CUDA GEMM 算子的一般实现与优化思路是什么？

1. **合并访存与向量化。** 相邻 lane 尽量访问连续地址，减少内存事务；`float4` 等向量化减少单线程搬运指令，两者都需要满足布局、对齐和边界条件。
2. **分块复用。** 每个 block 负责一个输出 tile，将 A/B 的 K 分块搬入 shared memory；每线程再计算多个输出，通过寄存器外积复用 operand，并增加独立累加器。
3. **布局与流水线。** 用 warp tiling、padding/swizzle 匹配计算阶段的读取；预取下一 tile，让搬运与当前计算重叠。同步要保证“写完才读、读完才覆盖”，增加 stage 也会增加资源占用。
4. **按 shape 选参数。** 大 tile 提高复用，但增加寄存器/shared memory，可能降低驻留并行度或产生 spill；小 M/N、长 K 时可以评估 Split-K，但要计入归约成本。Tensor Core 路径需要重新匹配 fragment 布局。
5. **验证与融合。** 用同精度 cuBLAS/cuBLASLt 作基线，结合设备计时和 profiler 定位瓶颈；融合 bias、activation、residual 可减少中间访存，但要保持归约与数值语义。

扩展阅读：[CUDA GEMM 优化笔记](../../../../../gpu-kernel/GEMM-CUDA-Core优化.md)。

## 2. AMD RDNA 上 HIP WMMA GEMM 算子的实现与优化思路是什么？

1. **WMMA 与 MFMA。** RDNA 的 WMMA（Wave Matrix Multiply-Accumulate）和 CDNA 的 MFMA（Matrix Fused Multiply-Add）都是 wave 级矩阵指令。FlashRT 的 gfx1151 路径使用 BF16 `16×16×16`、wave32；MFMA 常用 wave64，有多种矩阵形状和精度，需按具体架构选择。
2. **小矩阵沿 K 分工。** FlashRT 的 `M=10,N=32,K=1024` 投影用四个 wave 各算 K=256，再在 LDS 合并 FP32 partial，占 4 KiB。这样增加组内并行，不需要 global atomic；输出 grid 仍只有两个 workgroup。
3. **Tiled 路径共享 A。** 每个 wave 负责不同的 16 列，协作搬入 A/W，复用同一 A tile。权重预转置让 K 方向连续；16-byte global 搬运、LDS padding 和 barrier 分别处理搬运效率、bank 映射与同步。当前 FlashRT tiled 路径是单缓冲。
4. **迁移到 CDNA 要重做布局与参数。** Wave32 改为 wave64 后，输入/累加器的 lane 映射、LDS 访问和输出写回都不同；还要按 MFMA 的吞吐、延迟和寄存器/LDS 预算重选 tile、wave 数与 stage 数。不能只替换指令或照搬 padding。
5. **调优与正确性。** FlyDSL 的工作是在已有 GEMM 上按 gfx1151 的设备规模和 shape 调整 tile policy。实际执行保留 hipBLASLt fallback，并验证完整 wave 参与、tail、BF16 舍入和端到端收益；不能把局部算子加速直接当作模型加速。

扩展阅读：[FlashRT WMMA 实现](../../../../../gpu-kernel/flashrt-rdna35/07-SmallM-WMMA-GEMM-Kernel.md)、[FlyDSL tile 策略](../../../../../gpu-kernel/flydsl/FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)、[CDNA MFMA 示例](../../../../../gpu-kernel/flydsl/03-四个示例读懂FlyDSL.md)。
