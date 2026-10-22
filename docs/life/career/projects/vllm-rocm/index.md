---
title: "vLLM ROCm 模型适配与优化"
---

# vLLM ROCm 模型适配与优化

从模型语义对齐进入量化与 LoRA 算子优化，再核对服务层收益。

1. [面试问题与回答](面试问题与回答.md)：十个核心问题，每题提供 1 分钟简答和 5 分钟详述，沿 CogAgent 开发链、缓存、数值、量化与 ROCm 优化展开，再讲 LoRA 和调度。
2. [CogAgent 模型接入与量化推理](01-CogAgent模型接入与量化推理.md)：双视觉路径、执行契约、RoPE 与 AWQ。
3. [LoRA 算子与优先级调度](02-LoRA算子与优先级调度.md)：Shrink/Expand、metadata、Graph 与请求调度。
4. [性能证据与面试问答](03-性能证据与面试问答.md)：测试配置、历史报告、增量对照与端到端归因。

原有报告副本及其来源、哈希与证据级别见 [evidence 清单](evidence/manifest.json)。

[返回项目导航](../index.md)
