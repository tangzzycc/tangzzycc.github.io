---
title: "GPU 算子实现"
hide:
  - toc
  - footer
---

# GPU 算子实现

按算子整理题目、给定签名、Tips、公共前置与分级实现。点击版本标签，查看对应说明和代码。

本栏目搬运自 [CUDA 手撕算子 · 面试练习](https://kkcocoa.github.io/cuda-operator-interview/)，以 2026-10-22 的页面内容为准，共 18 道题、85 个实现版本。代码中的挖空标记已去除，保留完整代码；第一篇沿用本站此前整理的样例。

## 访存基础

- [Fused Bias + GELU（向量化 elementwise）](<Fused Bias + GELU.md>)
- [矩阵转置（shared memory / bank conflict / swizzle）](<矩阵转置.md>)

## Reduce

- [Reduce Sum 七连击（归约优化全流程）](<Reduce Sum.md>)
- [warpReduce / blockReduce 模板工具箱](<Warp 与 Block 归约模板.md>)
- [Reduce Max / ArgMax（含 float atomicMax 技巧）](<Reduce Max 与 ArgMax.md>)

## Scan

- [Prefix Sum / Scan（Hillis-Steele → Blelloch → decoupled lookback）](<Prefix Sum 与 Scan.md>)

## Norm

- [Softmax（safe / online 一遍法 / 寄存器缓存）](<Softmax.md>)
- [LayerNorm（一趟 vs Welford 的数值取舍 + 融合 residual）](<LayerNorm.md>)
- [RMSNorm（LLaMA 形态：融合 residual + half IO）](<RMSNorm.md>)

## RoPE

- [RoPE 旋转位置编码（两种配对 + GQA + KV cache 偏移）](<RoPE.md>)

## Attention

- [朴素 Attention（三段式 → 融合，理解 O(S²) 的痛点）](<朴素 Attention.md>)
- [FlashAttention（分块 + online softmax + causal 跳块）](<FlashAttention.md>)
- [Decode Attention（KV cache / flash-decoding / paged / GQA 共享读）](<Decode Attention.md>)

## GEMM

- [SGEMM 逐级优化（naive → shared → register tile → 向量化 → 双缓冲 → wmma）](<SGEMM.md>)
- [GEMV / decode 算子（split-K + W8A16 反量化融合）](<GEMV.md>)

## TopK/Sort

- [TopK（迭代 / 局部堆 / warp bitonic / radix select）](<TopK.md>)
- [排序（bitonic sort + radix sort 的 1-bit split）](<排序.md>)

## 量化

- [量化 / 低精度（per-channel、dp4a、fp8 KV cache）](<量化与低精度.md>)

