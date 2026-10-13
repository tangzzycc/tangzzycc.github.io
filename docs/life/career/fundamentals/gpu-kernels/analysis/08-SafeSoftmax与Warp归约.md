---
title: "Safe Softmax：数值稳定、Warp 归约与寄存器暂存"
---

# Safe Softmax：数值稳定、Warp 归约与寄存器暂存

<div class="study-notes study-analysis" markdown="1">

来源：提供的「腾讯 AI Infra 部门二面」记录；对应 GPU与算子优化[第 43～45 题](../index.md#q33)。记录中的“Save Softmax”按上下文理解为 Safe Softmax，即数值稳定的 Softmax。

## 1. 减最大值保持结果，同时避免溢出 {#stable-softmax}

对含至少一个有限有效值的一行：

$$
m=\max_i x_i,\quad e_i=\exp(x_i-m),\quad
l=\sum_i e_i,\quad y_i=e_i/l.
$$

分子和分母同乘 `exp(−m)`，所以结果不变；有限有效输入的 `xᵢ−m≤0`，指数不超过 1。最大元素的指数为 1，避免所有有效项同时下溢为零。

例如 `[1000,1001,1002]` 直接指数容易溢出，减最大值后为 `[-2,-1,0]`，指数约 `[0.1353,0.3679,1]`，输出约 `[0.0900,0.2447,0.6652]`。

Mask 项可设为负无穷。全 mask 行的 `−∞−(−∞)` 会产生 NaN，需按算子契约单独处理；正无穷、NaN 的行为也要明确，不能由“减最大值”自动解决。

返回面试题：[43](../index.md#q33)
{.study-source}

## 2. 一行由一个 Warp 处理 {#warp-reduction}

设每 lane 处理 I 个元素，lane ℓ 访问 `ℓ+32j`，j 从 0 到 I−1；相邻 lane 访问相邻元素。

1. 加载并暂存输入，越界项填负无穷。
2. 每 lane 求自己的 max，再做 warp max all-reduce。
3. 原地生成指数并暂存，每 lane 求局部 sum，再做 warp sum all-reduce。
4. 用全行分母归一化，写出有效位置。

完整、未分歧的 32-lane warp 可用以下归约模式：

```cuda
// 所有 32 个 lane 执行；v 为每 lane 的局部值。
for (int delta = 16; delta > 0; delta >>= 1) {
    float other = __shfl_xor_sync(0xffffffffu, v, delta);
    v = fmaxf(v, other);  // sum 归约时改为 v += other
}
```

XOR 树结束后各 lane 都得到归约结果；若改为 down-shuffle 树，通常先在指定 lane 得到结果，再广播。`_sync` 的参与 mask 必须包含所有约定执行该指令的活跃 lane；不能让部分 lane 提前返回后仍用完整 mask。

最后一个不完整数据行仍可让完整 warp 参与：无有效数据的 lane 用 max 的负无穷、sum 的零，不是让这些 lane 消失。更一般的稀疏活跃 mask 需要另外设计归约算法。

### 多 warp 处理一行，第二轮归约怎样组织？

每线程先在寄存器中对多个元素求局部 max；每 warp 用 shuffle 合并，由 lane 0 写一份 warp max 到 shared。CTA 同步后，第一个 warp 读取这些部分值，其余 lane 使用 max 单位元 −∞，归约出整行 max，再写 shared 并同步广播。随后各线程生成指数，sum 重复同一层级流程，最后归一化写回。

CTA 的第二次同步保证所有线程读取到同一个行 max，不能由 warp 0 写完后让其他 warp 无同步读取。最后 warp 的无有效数据 lane 仍需按协议参与，sum 用 0、max 用 −∞，避免提前 return 破坏 block barrier。

**Shuffle 传递寄存器值，只对参与 warp 有效；它既不是跨 warp 通信，也不是 shared/global 内存的完成 barrier。**
{.study-caution}

返回面试题：[44](../index.md#q34)
{.study-source}

## 3. 寄存器暂存怎样降低流量？ {#register-staging}

用局部数组保存每 lane 的输入，求 max 后覆盖成指数，归约 sum 后直接归一化写回。在无 spill、输入输出各一次访问的理想条件下，元素数 N、输入输出各 s byte，主要 global 载荷约 `2Ns`，不含事务浪费与其他数据。

没有暂存时，分阶段求 max、求指数和 sum、写结果可能反复读取输入或写出中间指数。融合与暂存省去这些流量，但编译器是否真正放到寄存器、是否 spill，需要看资源报告和 profiler。

### 暂存多少值会触发资源问题，怎么判断是否真的在寄存器？

N 元素由 T 个线程处理时，每线程至少保存约 ceil(N/T) 个输入或指数，还要 max/sum、地址与中间值；编译器展开和变量生命周期决定真实寄存器数量。源代码写局部数组不保证编译到寄存器，动态索引或预算不足可能落到 local memory。

如果暂存使驻留 CTA 从 2 变 1，或出现 local load/store，省掉的输入读取可能被额外 spill 和更少独立 warp 抵消。应看编译资源报告与实际流量，再比较一次暂存与分块重读；具体长行选择见[行长度与两次扫描](08-SafeSoftmax与Warp归约.md#row-length)。

返回面试题：[45](../index.md#q35)
{.study-source}

## 4. 行长度改变执行选择 {#row-length}

| 情况 | 常见选择 | 要关注的成本 |
| --- | --- | --- |
| 较短行 | 单 warp、一 lane 多元素 | 并行行数、指数指令、提交开销 |
| 较长行 | 多 warp 的 block 协作 | 跨 warp shared 归约与同步、寄存器预算 |
| 很长行或并行度不足 | 多阶段或分块归约 | 中间状态、额外访问与归约 launch |

“较短/较长”的边界由 dtype、硬件、并发和寄存器预算决定。更大暂存不一定更快，指数计算与归约也可能主导。若采用 online softmax，分块 max/sum 需要按变化的最大值重缩放；这与 Safe Softmax 的稳定目标相关，但不是同一种执行组织。

### 很长一行为什么可能选择“重读输入”而不全部暂存？

一行长度 N、线程数 T，每线程保存约 ceil(N/T) 个值；暂存和展开很大时，寄存器会降低驻留或 spill 到 local，可能比再读一次连续输入更贵。对很长行可以先分块求局部统计，再合并全行 max/sum，最终重新读输入生成归一化输出。这里主动增加必需读取，换取更小寄存器与更多并行。

Online softmax 能在一次扫描中维护稳定 max/sum，但输出 yᵢ 需要全行最终分母。若没有保存全部输入或未归一化指数，仍要第二次扫描写最终输出；它不自动让独立 Softmax 像 FlashAttention 一样只保留小状态即可生成所有位置输出。FlashAttention 之所以可累计小输出向量，是其目标为 `PV`，不是要求物化全部 P。

## 5. 验证应覆盖什么？ {#correctness}

数值检查包括极大有限输入、相等输入、非 32 倍数行、mask 与全 mask 约定，对照输出和、非负性与稳定参考结果。性能检查覆盖真实行长度、dtype、批量，观察寄存器、spill、global 流量与正式计时；profiler 耗时不等于正常执行耗时。

资料：[CUDA Warp Shuffle Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions)。相关机制：[CP 的 Softmax 状态合并](../../inference/analysis/10-ContextParallel.md)。

</div>
