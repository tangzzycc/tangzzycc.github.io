---
title: "AI Infra"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">推理系统 · 工程实践</p>

# AI Infra

围绕大模型推理基础设施整理学习方向。从并行部署、推理框架到 GPU 算子，沿计算、通信与资源分配理解系统。

</div>

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">分布式推理</p>

### [推理并行策略](推理并行策略.md)

TP、PP、DP、EP、DP Attention、SP、CP：参数与数据怎样切分，在哪里通信，以及如何组合。正文按机制配局部图，[完整流程图附件](推理并行策略-完整流程图.md)收录七张高清长图。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">vLLM</p>

### [推理框架](../vllm/index.md)

请求生命周期、调度、KV Cache 与模型执行。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">GPU Kernel</p>

### [GPU 算子](../gpu-kernel/index.md)

从 GPU 执行模型到算子源码与性能优化。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">开源 PR 追踪</p>

### [开发与贡献](../pr-tracking/index.md)

沿实际 PR 理解问题、实现与验证过程。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

</div>
