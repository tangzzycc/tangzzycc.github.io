# Weight Dequantization：权重反量化

## 题目

编写一个 GPU 程序，对权重矩阵进行反量化。给定包含量化值、形状为 $[M,N]$ 的矩阵 $X$，以及形状为 $[\lceil M/T\rceil,\lceil N/T\rceil]$ 的缩放因子矩阵 $S$，其中 $T$ 为分块大小。每个元素 $X_{i,j}$ 对应的缩放因子为 $S_{row,col}$，其中 $row = \lfloor i/T\rfloor$、$col = \lfloor j/T\rfloor$，输出为：

$$
Y_{i,j} = X_{i,j} \times S_{row, col}
$$

## 实现
