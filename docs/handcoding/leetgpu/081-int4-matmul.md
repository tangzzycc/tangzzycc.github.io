# INT4 Weight-Only Quantized MatMul：W4A16 量化矩阵乘法

## 题目

编写一个 GPU 程序，实现仅对权重量化的 INT4 矩阵乘法（W4A16）。给定形状为 $M \times K$ 的 FP16 激活矩阵 $x$ 和以打包 INT4 格式存储的权重矩阵，计算形状为 $M \times N$ 的输出矩阵 $y$：

$$
y = x \times W^T.
$$

其中 $W$ 是形状为 $N \times K$ 的反量化 FP16 权重矩阵。

$w_q$ 的每个字节存储两个 INT4 权重：高 4 位（Bits 7–4）存储 $w[n,2i]$，低 4 位（Bits 3–0）存储 $w[n,2i+1]$。INT4 值以 $[0,15]$ 的无符号形式存储，减去偏移量 $8$ 后得到 $[-8,7]$ 的有符号权重。

权重按组反量化：沿 $K$ 维，每连续 $\mathrm{group\_size}$ 个权重共享一个 FP16 缩放因子。令 $\mathrm{w\_q\_nibble}[n,k]$ 为解包后的 4 位无符号值，则：

$$
W[n,k] =
\left(\mathrm{w\_q\_nibble}[n,k]-8\right)
\times \mathrm{scales}\left[n,\left\lfloor k/\mathrm{group\_size}\right\rfloor\right].
$$

## 实现
