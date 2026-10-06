---
title: "LeetGPU 练习"
hide:
  - navigation
  - toc
  - footer
---

<div class="notebook-index" markdown="1">

<div class="index-intro" markdown="1">

<p class="index-eyebrow">HandCoding / GPU 算子 · 16 道题</p>

# LeetGPU 练习

按题目整理 GPU Kernel 的实现与优化。同一道题的不同实现放在同一篇中，从输入输出约定开始，再追踪计算与数据移动。

</div>

<div class="index-grid" markdown="1">

<section class="index-card" markdown="1">

<p class="index-card-label">题目 001</p>

### [Vector Addition：向量加法](001-vector-add.md)

逐元素向量加法与基本线程索引。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 002</p>

### [Matrix Multiplication：矩阵乘法](002-matrix-multiplication.md)

矩阵乘法的计算与内存访问。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 003</p>

### [Matrix Transpose：矩阵转置](003-matrix-transpose.md)

矩阵转置的读写映射与布局。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 004</p>

### [Reduction：求和归约](004-reduction.md)

并行求和与归约过程。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 005</p>

### [Softmax](005-softmax.md)

最大值归约、指数与归一化。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 006</p>

### [Softmax Attention](006-softmax-attention.md)

QK、Softmax 与 V 加权组合。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 016</p>

### [Prefix Sum：前缀和](016-prefix-sum.md)

前缀累加与线程之间的数据依赖。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 022</p>

### [General Matrix Multiplication (GEMM)：通用矩阵乘法](022-gemm.md)

带缩放与累加项的矩阵乘法。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 029</p>

### [Top K Selection：Top-K 选择](029-top-k-selection.md)

选择最大的 K 个元素。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 032</p>

### [INT8 Quantized MatMul：INT8 量化矩阵乘法](032-int8-quantized-matmul.md)

INT8 量化输入的矩阵乘法。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 053</p>

### [Causal Self-Attention：因果自注意力](053-causal-self-attention.md)

因果掩码与自注意力计算。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 061</p>

### [Rotary Positional Embedding：旋转位置编码](061-rope-embedding.md)

位置编码的旋转与元素配对。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 064</p>

### [Weight Dequantization：权重反量化](064-weight-dequantization.md)

从量化权重恢复数值。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 081</p>

### [INT4 Weight-Only Quantized MatMul：W4A16 量化矩阵乘法](081-int4-matmul.md)

INT4 权重与 FP16 输入的矩阵乘法。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 083</p>

### [Fused Residual Add and RMS Norm](083-fused-residual-add-rms-norm.md)

残差加法与 RMS Norm 融合。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

<section class="index-card" markdown="1">

<p class="index-card-label">题目 113</p>

### [Layer Normalization：层归一化](113-layer-normalization.md)

均值、方差与 Layer Norm。

<span class="index-card-arrow" aria-hidden="true">↗</span>

</section>

</div>

<details class="index-details" markdown="1">
<summary>题目来源与笔记约定</summary>

这里按题目整理 [LeetGPU](https://leetgpu.com) 的 Kernel 练习笔记。每篇先简要描述题目，再围绕具体实现展开，沿用本站已有算子笔记的写法。题目描述、变量与公式沿用英文题面，原题示意图按本站风格重绘，保留数据与关系。

题目来源为本地 `~/leetgpu-challenges` 仓库，对应 [tangzzycc/leetgpu-challenges](https://github.com/tangzzycc/leetgpu-challenges)。

每道题对应本目录下一个 Markdown 文件，标题只保留算子名称，文件名按原仓库题号命名，例如 `001-vector-add.md`、`022-gemm.md`。同一题的不同实现和优化过程放在同一篇里，标题随实现内容自然划分。

题目资料位于 `challenges/<difficulty>/<number>_<name>/`：`challenge.html` 是英文题面，`challenge.py` 包含参考实现与测试，`starter/` 提供各框架的入口代码。

</details>

<div class="index-more" markdown="1">

[返回 HandCoding](../index.md) · [LeetGPU](https://leetgpu.com) · [题目仓库](https://github.com/tangzzycc/leetgpu-challenges)

</div>

</div>
