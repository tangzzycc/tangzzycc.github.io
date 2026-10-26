---
title: "项目经历"
---

# 项目经历

围绕简历项目准备可展开的回答：先说明触发问题与个人工作，再解释实现、取舍和验证。性能数字对应具体模型、设备、负载与基线，按各项目的证据归档核对。

| 项目 | 复习重点 | 入口 |
| --- | --- | --- |
| vLLM ROCm 模型适配与优化 | CogAgent 接入、数值对齐、量化与视觉优化、LoRA Shrink/Expand 算子 | [面试问答](vllm-rocm/面试问题与回答.md) · [项目导航](vllm-rocm/index.md) |
| vLLM P/D 解耦与调度优化 | RAPID-Serve、V1/MRV2、资源共享与性能取舍 | [面试问答](vllm-pd/面试问题与回答.md) · [项目导航](vllm-pd/index.md) |
| Adaptive KV Offload | 分层缓存、重算成本、异步生命周期 | [项目导航](vllm-adaptive-kv/index.md) |

开发实现与 PR 贡献单独见[开发实践导航](../development/index.md)。

## 从基础知识进入项目案例

- **GEMM 与硬件布局：** [FlashRT PR](../development/flashrt/FlashRT-PR.md)、[FlyDSL PR](../development/flydsl/FlyDSL-PR.md)。
- **模型适配、量化与 LoRA：** [ROCm 实现](vllm-rocm/01-CogAgent模型接入与量化推理.md)、[LoRA 算子](vllm-rocm/02-LoRA算子与优先级调度.md)。
- **调度、KV 与异步执行：** [P/D 实现](vllm-pd/02-PD解耦项目详解.md)、[Adaptive KV 实现](vllm-adaptive-kv/02-调度接入与异步生命周期.md)。

[基础知识复习](../fundamentals/index.md) · [简历](../resume/index.md)
