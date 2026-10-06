# INT8 Quantized MatMul：INT8 量化矩阵乘法

## 题目

编写一个 GPU 程序，对两个 8 位有符号整数矩阵进行量化矩阵乘法。给定形状为 $M \times K$ 的矩阵 $A$、形状为 $K \times N$ 的矩阵 $B$，输入量化缩放因子 $\mathrm{scale\_A}$、$\mathrm{scale\_B}$，输出缩放因子 $\mathrm{scale\_C}$，以及零点 $\mathrm{zero\_point\_A}$、$\mathrm{zero\_point\_B}$、$\mathrm{zero\_point\_C}$，计算：

$$
C_{\text{quant}}(i, j) = \mathrm{clamp}\left(
         \mathrm{round}\left(
             \frac{
                 \sum_{k=0}^{K-1} (A_{ik} - z_A)(B_{kj} - z_B) \cdot s_A s_B
             }{s_C}
         \right) + z_C,\ -128,\ 127
     \right)
$$

其中 $s_A = \mathrm{scale\_A}$、$z_A = \mathrm{zero\_point\_A}$，$B$ 和 $C$ 的符号对应关系相同。扣除输入零点后以 INT32 累加，再以 FP32 缩放，舍入到最近整数，加上输出零点并截断到 $[-128,127]$，最终以 INT8 写入 $C$。

## 实现
