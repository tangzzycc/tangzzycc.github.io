---
title: "AI Infra 面试基础"
---

# AI Infra 面试基础

面向秋招 AI Infra 方向，按 GPU与算子优化、推理优化、模型结构三个模块复习。

每个模块的 `index.md` 默认只展示问题，点击展开面试示例回答，核心重点用粗体标注；`analysis/` 下按主题保留具体分析、算例、源码与追问。示例回答通常为一段，约 30～60 秒。

| 模块 | 问题与示例回答 | 具体分析 |
| --- | --- | --- |
| GPU与算子优化 | [5 道问答](gpu-kernels/index.md) | [内存层次](gpu-kernels/analysis/01-内存层次.md)、[MI355X](gpu-kernels/analysis/02-MI355X硬件参数.md)、[GEMM](gpu-kernels/analysis/03-GEMM实现与优化.md)、[GPU DSL](gpu-kernels/analysis/04-GPU-DSL.md) |
| 推理优化 | [50 道问答](inference/index.md) | [Graph 与 Compile](inference/analysis/01-Graph与Compile.md)、[vLLM 分析与源码版本](inference/analysis/index.md) |
| 模型结构 | [1 道问答](models/index.md) | [DeepSeek-V3](models/analysis/01-DeepSeek-V3.md) |

[项目与开源贡献](../projects/index.md)用于把上述知识接到自己的实现与性能证据。
