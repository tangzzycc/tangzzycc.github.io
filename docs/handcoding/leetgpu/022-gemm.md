# General Matrix Multiplication (GEMM)：通用矩阵乘法

## 题目

编写一个 GPU 程序，实现通用矩阵乘法（GEMM）。给定形状为 $M \times K$ 的矩阵 $A$、形状为 $K \times N$ 的矩阵 $B$、形状为 $M \times N$ 的输入/输出矩阵 $C$，以及标量乘数 $\alpha$ 和 $\beta$，计算：

$$
C = \alpha \cdot (A \times B) + \beta \cdot C_{initial}
$$

$A$、$B$ 和 $C$ 的初始值均为 FP16，所有矩阵按行优先存储，$\alpha$ 和 $\beta$ 为 FP32 标量。乘法累加使用 FP32，最终结果转换为 FP16 并写回 $C$。

## 实现
