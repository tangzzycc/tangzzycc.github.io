---
title: "开源贡献面试准备"
---

# 开源贡献面试准备

按仓库复习个人贡献：先完整介绍工作，再沿触发问题、调用链与设计难点连续展开。面试笔记直接展示正文，完整 PR 材料保留实现细节与实验记录。

- [vLLM 面试问题与回答](vLLM-面试问题与回答.md)：完整贡献介绍，再从 8 个 PR 连续展开改动、调用链与基础追问。
- [vLLM PR 深入材料](vLLM-PR.md)：多模态 Attention、MRV2、投机解码、量化与 ROCm 分发的完整分析与实验记录。
- [FlashRT 面试问题与回答](FlashRT-面试问题与回答.md)：沿 Pi0.5 pipeline 复习初始化、Prefix、动作去噪、HIP Graph 与 Temporal KV 的优化点。
- [FlashRT PR 深入材料](FlashRT-PR.md)：Pi0.5 的 HIP 后端、Attention、WMMA GEMM、Graph 与 Temporal KV 的实现分析与实验记录。
- [FlyDSL PR](FlyDSL-PR.md)：GEMM tile policy、JIT/cache 与 RoPE 布局修复。
- [FlyDSL RoPE 修复口述](FlyDSL-RoPE修复.md)：D=96 的向量宽度与 lane 配对，保留简短回答与四个分析点。

通用知识另见[基础复习](../../fundamentals/index.md)。

[返回项目导航](../index.md)
