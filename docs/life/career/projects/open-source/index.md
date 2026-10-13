---
title: "开源贡献面试准备"
---

# 开源贡献面试准备

按仓库复习个人贡献：触发问题、修改位置、实现机制与验证证据。完整 PR 材料保留已有深入追问。

- [vLLM PR](vLLM-PR.md)：多模态 Attention、MRV2、投机解码、量化与 ROCm 分发。
- [FlashRT PR](FlashRT-PR.md)：Pi0.5 的 HIP 后端、Attention、WMMA GEMM、Graph 与 Temporal KV。
- [FlyDSL PR](FlyDSL-PR.md)：GEMM tile policy、JIT/cache 与 RoPE 布局修复。
- [FlyDSL RoPE 修复口述](FlyDSL-RoPE修复.md)：D=96 的向量宽度与 lane 配对，保留简短回答与四个分析点。

PR 追问涉及的框架机制可回查 [vLLM 框架问答](../../fundamentals/inference/index.md)，DSL 与矩阵计算可回查 [基础复习](../../fundamentals/index.md)。

[返回项目导航](../index.md)
