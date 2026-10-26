---
title: "vLLM ROCm 模型适配与优化"
---

# vLLM ROCm 模型适配与优化

从 CogAgent 模型接入、计算与缓存进入数值对齐和 ROCm 优化，再单独复习 LoRA Shrink/Expand 算子与服务层收益。

1. [面试问题与回答](面试问题与回答.md)：五个主题、十二道问答，含可跳转概览、项目口述、张量形状、两类 KV、BF16、AWQ、视觉执行和 LoRA 算子追问。
2. [CogAgent 模型接入与量化推理](01-CogAgent模型接入与量化推理.md)：双视觉路径、执行契约、RoPE 与 AWQ。
3. [LoRA 算子详解](02-LoRA算子与优先级调度.md#lora-math)：低秩计算、Prefill/Decode kernel、adapter metadata 与 Graph 正确性。
4. [性能证据与面试问答](03-性能证据与面试问答.md)：测试配置、历史报告、增量对照与端到端归因。

原有报告副本及其来源、哈希与证据级别见 [evidence 清单](evidence/manifest.json)。

[返回项目导航](../index.md)
