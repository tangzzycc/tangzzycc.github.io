---
title: "FlashRT RDNA 3.5"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">GPU Kernel / HIP · Wave32 · BF16</p>

# FlashRT RDNA 3.5

围绕 FlashRT PR #213 的 Pi0.5 后端，从硬件与推理流程进入 24 个 RDNA HIP Kernel，理解数据所有权、访存、归约与算子分发。

</div>

## 硬件与推理流程

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [AMD GPU 硬件模型与 Kernel 阅读术语](00-AMD-GPU硬件模型与术语.md)

GPU 执行层级、数据所有权与 VGPR / SGPR / LDS / Cache。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [Frontend 与 Pipeline](01-Pi05-Frontend与Pipeline.md)

从 OpenPI 模型到静态推理流程与图捕获。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03</p>

### [Pi0.5 模型结构与推理流程](02-Pi05模型结构与推理流程.md)

从图像、指令与随机噪声生成连续动作。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## Kernel 实现

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">04</p>

### [HIP 执行模型与 Wave32 归约](01-HIP执行模型与Wave32归约.md)

Launch、Thread Ownership 与 Wave32 / Block Reduction。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">05</p>

### [Activation 与 Residual Kernel](02-Activation与Residual-Kernel.md)

Scalar / Packed 路径、Merged GeGLU 与 Gated Residual。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">06</p>

### [Normalization Kernel](03-Normalization-Kernel.md)

8 个 Norm Kernel 的行内归约与寄存器复用。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">07</p>

### [QKV Split 与 RoPE Kernel](04-QKV-Split与RoPE-Kernel.md)

Complex Pair 所有权与 8Q / 1K / 1V 的 Cos / Sin 复用。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">08</p>

### [Encoder FlashAttention Kernel](05-Encoder-FlashAttention-Kernel.md)

Wave Pair 协作、Online Softmax 与 QK / PV 计算。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">09</p>

### [Decoder GQA Kernel](06-Decoder-GQA-Kernel.md)

Split-key 划分与 Row-owned Decode 实现。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">10</p>

### [Small-M WMMA GEMM Kernel](07-SmallM-WMMA-GEMM-Kernel.md)

WMMA Mapping、Split-K、LDS Tiling 与 hipBLASLt 分工。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## 跨帧优化

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">11</p>

### [Temporal K/V 跨帧复用](06-Temporal-KV跨帧复用.md)

跨帧 Prefix K/V 复用、失败恢复与图刷新。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

<details class="index-details" markdown="1">
<summary>源码版本与阅读约定</summary>

- PR 合并后固定版本：[`e2cf5d1`](https://github.com/flashrt-project/FlashRT/commit/e2cf5d1fe7209e3f4fbf212c5b6321467e8f4874)
- PR 基线：[`dd549e2`](https://github.com/flashrt-project/FlashRT/commit/dd549e2)
- 目标硬件：AMD RDNA 3.5，`gfx1151`，Wave32
- 数值路径：BF16 Input/Output，FP32 Accumulation

> 正文代码块中的中文注释由本系列补充，用于标记 Ownership、索引单位、数据生命周期与同步边界，不属于上游源码原注释。未加说明的代码仍对应固定提交中的实现；代码块中若出现 `...`，表示为了聚焦当前机制而明确省略的源码。

</details>

<details class="index-details" markdown="1">
<summary>24 个 Kernel 清单与验证边界</summary>

PR 新增了 24 个 `__global__` Kernel body。这里按源码定义计数；Template 可能在编译时实例化出多个不同的 Machine Kernel。

| 实现族 | `__global__` Kernel | 数量 |
| --- | --- | ---: |
| Activation | `gelu_rdna_kernel`、`gelu_packed_rdna_kernel`、`gelu_mul_rdna_kernel`、`gelu_mul_packed_rdna_kernel`、`gelu_mul_merged_rdna_kernel`、`gelu_mul_merged_packed_rdna_kernel`、`silu_rdna_kernel`、`silu_packed_rdna_kernel` | 8 |
| Residual | `residual_scalar_rdna_kernel`、`residual_rdna_kernel<HasGate>` | 2 |
| Normalization | `layer_norm_wide_rdna_kernel`、`rms_norm_wide_rdna_kernel`、`residual_rms_wide_rdna_kernel`、`layer_norm_rdna_kernel`、`rms_norm_rdna_kernel`、`adarms_rdna_kernel`、`residual_rms_rdna_kernel`、`residual_adarms_rdna_kernel` | 8 |
| RoPE | `qkv_rope_rdna_kernel` | 1 |
| Encoder Attention | `attention_encoder_gqa_rdna_kernel` | 1 |
| Decoder Attention | `attention_decoder_gqa_splitkey_rdna_kernel`、`attention_decoder_gqa_rdna_kernel` | 2 |
| Small-M GEMM | `smallm_wmma_bf16_tiled_rdna_kernel`、`smallm_wmma_bf16_rdna_kernel` | 2 |

`hipblaslt_runner_rdna.hip` 不定义 `__global__` Function，而是通过 hipBLASLt 调用 Library Kernel。它仍然会在第 07 章中讲解，因为只有明确通用 GEMM 与手写 Small-M Kernel 的分界，才能理解 PR 为什么不用 WMMA Kernel 接管所有 Linear。

### 每个 Kernel 都用同一组问题阅读

后续章节不会只把 C++ 代码改写成中文。对每个 Kernel，都会明确回答：

1. 输入、输出和 Shape Contract 是什么。
2. `gridDim`、`blockDim`、`blockIdx`、`threadIdx`、Wave 和 Lane 分别表示什么。
3. 一个 Thread/Wave/Workgroup 拥有哪些 Value，覆盖是否无重叠且无遗漏。
4. 每次 Global Memory 或 LDS 访问有多宽，是否连续、对齐、需要 Tail Guard。
5. 哪些 Value 位于 Register，哪些位于 LDS，什么时候必须 Barrier。
6. FP32 Accumulation 和 BF16 Cast 发生在哪里，融合前后是否保留同一 Rounding Boundary。
7. Host Launcher 与 Python Provider 在什么条件下选择它，什么时候走 Fallback。

每篇正文会先列出本章覆盖的 `__global__` Kernel、相关 Host Launcher/Helper 和推荐阅读入口，再把固定提交中的关键 Kernel Body 直接放进笔记，贴着源码解释实现。源码阅读顺序默认是 **Launch 与 Shape Contract → Index/Ownership → GMEM/LDS/Register Dataflow → Barrier/Reduction → Epilogue/Rounding → Provider Dispatch**；这样不需要在笔记和仓库之间频繁切换，也能解释 Kernel 为何只在特定 Shape 下出现。

这种读法会把“算子的数学定义”与“GPU 上的物理执行”分开，再通过数据所有权把两者接回来。比如 `GELU(x)` 的数学式没有 Thread，但 `gelu_packed_rdna_kernel` 的物理实现是每个 Thread 拥有一个 BF16x2；Attention 的数学式只有 $QK^T$ 和 Softmax，但 Encoder Kernel 中的真实单位是协作的 Wave Pair。

### 源码与验证边界

本系列将 PR 合并后提交 `e2cf5d1` 作为唯一实现依据，不用当前 `main` 或后续 PR 的代码补齐细节。每篇会同时检查 HIP Function、Host Dispatch、Python Provider 与对应 Test，因为单独看 `__global__` Function 无法得知它在生产路径中是否会被选中。

算子测试能证明指定 Shape 与容差下的 HIP/PyTorch 对齐，但它不能自动证明“所有 Shape 都适用”或“所有机器上都更快”。本系列因此只说明已经写入源码的 Contract、Dispatch 和数值语义，不把某组开发机 Benchmark 外推为通用性能结论。

</details>

<div class="index-more" markdown="1">

[返回 GPU Kernel](../index.md) · [FlashRT PR #213](https://github.com/flashrt-project/FlashRT/pull/213)

</div>

</div>
