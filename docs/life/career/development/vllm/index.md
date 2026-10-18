---
title: "vLLM 开发实践与贡献"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">开发实践 / vLLM / ROCm</p>

# vLLM 开发实践与贡献

沿模型接入、ROCm 算子、Attention 边界与投机状态修复整理实现分析，再结合 PR 材料与面试问答复习个人工作。

[面试问题与回答](vLLM-面试问题与回答.md) · [PR 材料](vLLM-PR.md) · [vLLM 框架笔记](<../../../../vllm/基础篇/vLLM V1整体架构.md>)

</div>

## 实现分析 <span class="index-section-count">10 篇</span> { #analysis }

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [CogAgent-18B 在 vLLM 中的模型接入与推理优化](CogAgent-18B在vLLM中的模型接入与推理优化.md)

CogAgent-18B 的模型接入、视觉路径与推理优化。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [RDNA 3.5 上的 vLLM LoRA Kernel 优化](RDNA3.5上的vLLM-LoRA-Kernel优化.md)

RDNA 3.5 上的 LoRA 算子实现与分发。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03 · PR #53283</p>

### [单输出 GEMM 与 Shared Expert 门控的 ROCm 优化](单输出GEMM与Shared-Expert门控的ROCm优化.md)

单输出 GEMM 与 Shared Expert 门控的 ROCm 路径。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">04 · PR #52619</p>

### [RDNA W4A16 的形状分发与 LDS 边界](RDNA-W4A16的形状分发与LDS边界.md)

W4A16 的形状分发与 LDS 使用边界。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">05 · PR #52637</p>

### [GPTQ 权重加载与 RDNA INT4 布局重排](GPTQ权重加载与RDNA-INT4布局重排.md)

GPTQ 权重加载与 RDNA INT4 的布局重排。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">06 · PR #58513</p>

### [Channelwise GPTQ MoE 的加载与 Triton 执行](Channelwise-GPTQ-MoE的加载与Triton执行.md)

Channelwise GPTQ MoE 的权重加载与 Triton 执行。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">07 · PR #59550</p>

### [ROCm Attention 滑动窗口的参数与边界](ROCm-Attention滑动窗口的参数与边界.md)

滑动窗口的参数约定与边界处理。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">08 · PR #60040</p>

### [多模态 Prefix-LM 在 ROCm Attention 中的实现](多模态Prefix-LM在ROCm-Attention中的实现.md)

图像 token 双向注意力与 Prefix-LM 的实现。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">09 · PR #59600</p>

### [MRV2 投机解码中的 ShortConv 历史恢复](MRV2投机解码中的ShortConv历史恢复.md)

投机解码轮次之间的 ShortConv 历史恢复。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">10 · PR #60049</p>

### [MRV2 的 PIECEWISE 填充与投机解码启动 Profiling](MRV2的PIECEWISE填充与投机解码启动Profiling.md)

PIECEWISE 填充与启动 Profiling 的执行边界。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

[返回开发实践导航](../index.md)

</div>
