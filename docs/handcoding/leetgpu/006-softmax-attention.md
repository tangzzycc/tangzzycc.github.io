# Softmax Attention

## 题目

编写一个 GPU 程序，计算 Softmax Attention。给定形状为 $M \times d$ 的查询矩阵 $Q$、形状为 $N \times d$ 的键矩阵 $K$ 和形状为 $N \times d$ 的值矩阵 $V$，按下式计算输出矩阵，其中 Softmax 逐行计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\Bigl( \frac{QK^T}{\sqrt{d}} \Bigr)V,
$$

## 实现
