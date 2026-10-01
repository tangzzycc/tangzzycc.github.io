# LeetGPU Kernel 练习

这里按题目整理 [LeetGPU](https://leetgpu.com) 的 Kernel 练习笔记。每篇先简要描述题目，再围绕具体实现展开，沿用本站已有算子笔记的写法。题目描述、变量与公式沿用英文题面，原题示意图按本站风格重绘，保留数据与关系。

题目来源为本地 `~/leetgpu-challenges` 仓库，对应 [tangzzycc/leetgpu-challenges](https://github.com/tangzzycc/leetgpu-challenges)。

每道题对应本目录下一个 Markdown 文件，标题只保留算子名称，文件名按原仓库题号命名，例如 `001-vector-add.md`、`022-gemm.md`。同一题的不同实现和优化过程放在同一篇里，标题随实现内容自然划分。

题目资料位于 `challenges/<difficulty>/<number>_<name>/`：`challenge.html` 是英文题面，`challenge.py` 包含参考实现与测试，`starter/` 提供各框架的入口代码。

## 题目索引

- [Vector Addition：向量加法](001-vector-add.md)
- [Matrix Multiplication：矩阵乘法](002-matrix-multiplication.md)
- [Matrix Transpose：矩阵转置](003-matrix-transpose.md)
- [Reduction：求和归约](004-reduction.md)
- [Softmax](005-softmax.md)
- [Softmax Attention](006-softmax-attention.md)
- [Prefix Sum：前缀和](016-prefix-sum.md)
- [General Matrix Multiplication (GEMM)：通用矩阵乘法](022-gemm.md)
- [Top K Selection：Top-K 选择](029-top-k-selection.md)
- [INT8 Quantized MatMul：INT8 量化矩阵乘法](032-int8-quantized-matmul.md)
- [Causal Self-Attention：因果自注意力](053-causal-self-attention.md)
- [Rotary Positional Embedding：旋转位置编码](061-rope-embedding.md)
- [Weight Dequantization：权重反量化](064-weight-dequantization.md)
- [INT4 Weight-Only Quantized MatMul：W4A16 量化矩阵乘法](081-int4-matmul.md)
- [Fused Residual Add and RMS Norm](083-fused-residual-add-rms-norm.md)
- [Layer Normalization：层归一化](113-layer-normalization.md)

<!-- 新增笔记时在这里添加题目链接，并同步更新 mkdocs.yml 中的 LeetGPU 练习导航。 -->
