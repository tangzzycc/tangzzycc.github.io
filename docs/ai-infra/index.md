---
title: "AI Infra"
hide:
  - navigation
  - toc
  - footer
---

# AI Infra

本目录记录大模型推理架构、并行策略、vLLM 执行机制与 GPU 算子优化。

## 推理架构与并行

- [PD 分离推理架构详解](PD分离推理架构详解.md)：Prefill 与 Decode 的计算差异、性能干扰，以及分离部署和 KV Cache 传输机制。
- [推理并行策略](推理并行策略.md)：TP、PP、DP、EP、DP Attention、SP、CP 的切分方式、通信位置与适用场景。

## vLLM 执行机制

- [vLLM Multi-LoRA 推理](vLLM Multi-LoRA推理.md)：LoRA 权重管理、请求与 adapter 的映射，以及 Shrink / Expand 算子的执行过程。
- [vLLM Model Runner V2 架构](<vLLM Model Runner V2 架构.md>)：Persistent Batch 解耦、RequestState、异步调度与投机解码。

## GPU 算子优化

- [CUDA 矩阵乘法优化：从 Naive 到 cuBLAS](CUDA矩阵乘法优化：从Naive到cuBLAS.md)：Simon Boehm 的 CUDA SGEMM 优化文章完整译文，从朴素实现逐步优化到 Warp 分块。
