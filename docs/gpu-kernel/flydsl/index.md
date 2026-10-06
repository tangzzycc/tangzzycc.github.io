---
title: "FlyDSL"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">GPU Kernel / Python DSL · AMD GPU</p>

# FlyDSL

从编程模型与编译流程开始，理解 Layout、Thread / Value 所有权与数据移动，再沿四个官方示例阅读完整实现。

</div>

## 编程模型与示例

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">01</p>

### [FlyDSL 编程模型与编译流程](01-FlyDSL编程模型与编译流程.md)

GPU 执行层级、JIT / Kernel 边界、参数特化与 MLIR Lowering。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">02</p>

### [Layout 代数与 Tile 编程](02-Layout代数与Tile编程.md)

Shape / Stride、Divide、Slice、Tensor View 与 Tiled Copy。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">03 · 四个官方示例</p>

### [FlyDSL — 一次源码精读](03-四个示例读懂FlyDSL.md)

沿 Vector Add、Tiled Copy、TiledMma 与 Preshuffle GEMM 阅读源码。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

## PR 实践

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">PR #1057</p>

### [RDNA3 GEMM Tile Policy](FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md)

RDNA3 GEMM 的 Tile Policy 与形状选择。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">PR #1113</p>

### [Fused RoPE · Head-Dim 96](FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)

Head-Dim 96 的融合 RoPE 实现与数据布局。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

<details class="index-details" markdown="1">
<summary>源码版本与原文来源</summary>

前两篇以 FlyDSL `1941889400621f1fc3d8f03bada1a4e7380cdf7e` 为固定版本核对源码。第三篇经作者授权迁移 JinnPan 的《FlyDSL — 一次源码精读》，对应博客源码提交 `f1054a225ef7861efe99b60748a3697c40500d8c`，保留原文内容与插图。

</details>

<div class="index-more" markdown="1">

[返回 GPU Kernel](../index.md)

</div>

</div>
