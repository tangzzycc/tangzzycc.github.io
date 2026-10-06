---
title: "GPU Kernel"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">线程分工 · 内存层级 · 数据复用</p>

# GPU Kernel

从硬件执行方式出发，阅读算子实现与性能优化。关注数据由谁持有、如何移动，以及计算与同步如何配合。

</div>

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">HIP · Wave32 · BF16</p>

### [FlashRT RDNA 3.5](flashrt-rdna35/index.md)

从 Pi0.5 推理流程到 24 个 RDNA HIP Kernel，追踪 Launcher、Provider 与 Fallback。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">Python DSL · AMD GPU</p>

### [FlyDSL](flydsl/index.md)

理解编译边界与 Layout / Tile 编程，阅读四个官方示例和 PR 实践。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">CUDA · GEMM</p>

### [GEMM CUDA Core 优化](GEMM-CUDA-Core优化.md)

沿 Naive GEMM 到接近 cuBLAS 的过程，理解访存、分块与寄存器复用。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

<div class="index-more" markdown="1">

[HandCoding · LeetGPU 练习](../handcoding/leetgpu/index.md) · [GEMM Kernel 源码](https://github.com/tangzzycc/notebook/tree/master/docs/gpu-kernel/gemm-kernels)

</div>

</div>
