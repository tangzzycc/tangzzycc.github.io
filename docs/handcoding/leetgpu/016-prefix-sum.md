# Prefix Sum：前缀和

## 题目

编写一个 GPU 程序，对包含 32 位浮点数（FP32）的数组计算前缀和（累积和）。输入数组为 $[a,b,c,d,\ldots]$ 时，输出为 $[a,a+b,a+b+c,a+b+c+d,\ldots]$，即每个位置包含从数组起点到当前位置的元素之和。

![Prefix Sum 题目示意图](../../../assets/gpu-kernel/leetgpu/16-prefix-sum.png){ width="50%" }

## 实现
