# Rotary Positional Embedding：旋转位置编码

## 题目

编写一个 GPU 程序，对一批查询向量计算旋转位置编码（RoPE）。RoPE 使用预先计算的余弦和正弦分量旋转 Query 和 Key 向量，以编码位置信息。给定查询向量 $x$ 及对应的余弦和正弦向量，计算：

$$
\text{RoPE}(x) = x \odot \cos + \text{rotate\_half}(x) \odot \sin
$$

其中 $\odot$ 表示逐元素乘法。$\text{rotate\_half}(x)$ 交换向量的前后两半，并将原后半部分取负。对于维度为 $d$ 的向量：

$$
\text{rotate\_half}([x_1, \dots, x_{d/2}, x_{d/2+1}, \dots, x_d]) = [-x_{d/2+1}, \dots, -x_d, x_1, \dots, x_{d/2}]
$$

本题输入 $Q$、$\cos$ 和 $\sin$ 的形状均为 $(M,D)$，$M$ 为 Token 数，$D$ 为偶数的 Head 维度。余弦表和正弦表采用 Half-split 布局：每行的第 $j$ 维与第 $j+D/2$ 维取值相同，输出形状仍为 $(M,D)$。

## 实现
