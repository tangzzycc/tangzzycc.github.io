---
title: "vLLM 笔记"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">请求生命周期 · Attention · 开发实践</p>

# vLLM 笔记

沿一条请求理解推理框架：从预处理、调度与 KV Cache，到 GPU 执行和输出，再深入 Attention 与实际开发中的问题。

<div class="index-jump" markdown="1">

[基础篇](#basics) · [进阶篇](#attention) · [开发实践](#practice) · [杂记](#misc)

</div>

</div>

## 基础篇 <span class="index-section-count">15 篇</span> { #basics }

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [vLLM V1 整体架构](基础篇/vLLM V1整体架构.md)

API Server、EngineCore 与 Worker 的职责和协作。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [请求的数据形态：从 Client 到 GPU](基础篇/请求的数据形态：从Client到GPU.md)

请求跨越 Client、Core 与 GPU 时的数据结构变化。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03</p>

### [请求预处理：Chat Template 与 Tokenization](基础篇/请求预处理：Chat Template与Tokenization.md)

Chat Template、分词与多模态预处理如何衔接。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">04</p>

### [EngineCore](基础篇/EngineCore.md)

EngineCore 的启动、进程组织与核心执行循环。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">05</p>

### [从 EngineCoreRequest 到 Request](基础篇/从EngineCoreRequest到Request.md)

Core 侧如何接收请求并建立调度使用的 Request。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">06</p>

### [KVCacheConfig](基础篇/KVCacheConfig.md)

理解 KV Cache 的规格、分组与配置结构。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">07</p>

### [KVCacheConfig 的产生](基础篇/KVCacheConfig的产生.md)

从模型需求与可用显存推导 KV Cache 配置。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">08</p>

### [KVCacheConfig 的 Core 侧消费：逻辑资源初始化](基础篇/KVCacheConfig的Core侧消费：逻辑资源初始化.md)

Core 侧如何初始化逻辑缓存资源。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">09</p>

### [KVCacheConfig 的 Worker 侧消费：GPU KV Cache 初始化](基础篇/KVCacheConfig的Worker侧消费：GPU KV Cache初始化.md)

Worker 侧如何分配与绑定 GPU KV Cache。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">10</p>

### [Continuous Batching：动态更新执行批次](基础篇/Continuous Batching：动态更新执行批次.md)

请求进入、退出与批次更新如何协同。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">11</p>

### [Chunked Prefill](基础篇/Chunked Prefill.md)

Prefill 如何分块并与 Decode 共享调度。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">12</p>

### [PD 分离与 KV 传输](基础篇/PD分离与KV传输.md)

Prefill / Decode 分离后的 KV 传输路径。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">13</p>

### [请求变成 GPU 输入](基础篇/请求变成GPU输入.md)

请求如何变成模型执行使用的张量与元数据。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">14</p>

### [Attention 的计算机理与 vLLM 实现](基础篇/Attention的计算机理与vLLM实现.md)

从 Attention 数学计算到 vLLM 算子实现。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">15</p>

### [从 Hidden States 到 Output Token](基础篇/从Hidden States到Output Token.md)

模型输出如何经过采样成为下一枚 token。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## 进阶篇 <span class="index-section-count">6 篇</span> { #attention }

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [FlashAttention v1：从完整矩阵到分块精确计算](进阶篇/FlashAttention：分块计算与Online Softmax.md)

分块精确计算与 Online Softmax 的推导。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [FlashAttention-2：并行划分与非矩阵运算优化](进阶篇/FlashAttention-2：线程块与Warp的工作划分.md)

线程块与 Warp 的划分如何提高并行效率。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03</p>

### [FlashAttention-3：异步搬运与计算流水线](进阶篇/FlashAttention-3：异步搬运与计算流水线.md)

异步数据搬运与计算流水线如何衔接。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">04</p>

### [Decode Attention：Paged KV、GQA 与 Split-KV](进阶篇/Decode Attention：Paged KV、GQA与Split-KV.md)

Paged KV、GQA 与 Split-KV 的 Decode 路径。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">05</p>

### [MLA：低秩压缩、矩阵吸收与解耦 RoPE](进阶篇/MLA：低秩压缩、矩阵吸收与解耦RoPE.md)

低秩压缩、矩阵吸收与解耦 RoPE。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">06</p>

### [稀疏 Attention：选择性访问与高效计算](进阶篇/稀疏Attention：选择性访问与高效计算.md)

选择性访问 KV 与稀疏计算的实现。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## 开发实践 <span class="index-section-count">10 篇</span> { #practice }

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [CogAgent-18B 在 vLLM 中的模型接入与推理优化](开发实践/CogAgent-18B在vLLM中的模型接入与推理优化.md)

CogAgent-18B 的模型接入、视觉路径与推理优化。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [RDNA 3.5 上的 vLLM LoRA Kernel 优化](开发实践/RDNA3.5上的vLLM-LoRA-Kernel优化.md)

RDNA 3.5 上的 LoRA 算子实现与分发。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03 · PR #53283</p>

### [单输出 GEMM 与 Shared Expert 门控的 ROCm 优化](开发实践/单输出GEMM与Shared-Expert门控的ROCm优化.md)

单输出 GEMM 与 Shared Expert 门控的 ROCm 路径。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">04 · PR #52619</p>

### [RDNA W4A16 的形状分发与 LDS 边界](开发实践/RDNA-W4A16的形状分发与LDS边界.md)

W4A16 的形状分发与 LDS 使用边界。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">05 · PR #52637</p>

### [GPTQ 权重加载与 RDNA INT4 布局重排](开发实践/GPTQ权重加载与RDNA-INT4布局重排.md)

GPTQ 权重加载与 RDNA INT4 的布局重排。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">06 · PR #58513</p>

### [Channelwise GPTQ MoE 的加载与 Triton 执行](开发实践/Channelwise-GPTQ-MoE的加载与Triton执行.md)

Channelwise GPTQ MoE 的权重加载与 Triton 执行。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">07 · PR #59550</p>

### [ROCm Attention 滑动窗口的参数与边界](开发实践/ROCm-Attention滑动窗口的参数与边界.md)

滑动窗口的参数约定与边界处理。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">08 · PR #60040</p>

### [多模态 Prefix-LM 在 ROCm Attention 中的实现](开发实践/多模态Prefix-LM在ROCm-Attention中的实现.md)

图像 token 双向注意力与 Prefix-LM 的实现。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">09 · PR #59600</p>

### [MRV2 投机解码中的 ShortConv 历史恢复](开发实践/MRV2投机解码中的ShortConv历史恢复.md)

投机解码轮次之间的 ShortConv 历史恢复。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">10 · PR #60049</p>

### [MRV2 的 PIECEWISE 填充与投机解码启动 Profiling](开发实践/MRV2的PIECEWISE填充与投机解码启动Profiling.md)

PIECEWISE 填充与启动 Profiling 的执行边界。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## 杂记 <span class="index-section-count">1 篇</span> { #misc }

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [vLLM Multi-LoRA 推理](杂记/vLLM Multi-LoRA推理.md)

Multi-LoRA 的推理流程、显存管理与 SGMV 算子。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

</div>
