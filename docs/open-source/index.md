# Kernel PR 深读

这一组笔记从真实 PR 出发，讨论低比特量化、GEMM dispatch、Triton elementwise kernel 与
wave-level 数据映射。每篇都固定到一个公开 commit，用同一个问题串起数学语义、线程分工、
数据布局、fallback 和性能证据；图中展示机制，正文解释为什么这样设计。

## Comfy-Kitchen

| 笔记 | 核心问题 | 关键结论 |
| --- | --- | --- |
| [PR #127：Rowwise INT8 的 16-byte 向量化](Comfy-Kitchen-127-Rowwise-INT8-Vectorized-Load.md) | 怎样在不改量化语义和归约结构的前提下加速两遍扫描？ | 用严格 alignment/K guard 保护 typed `uint4` load，并保留 scalar fallback。 |
| [PR #129：16-lane 协作的 SVDQuant INT4 Quantizer](Comfy-Kitchen-129-SVDQuant-INT4-Quantizer.md) | 怎样让一个 64-element group 从单线程串行变成 wave 内协作？ | 每 lane 缓存 4 个值，width-16 shuffle 求 scale，一次写出 4 个 nibble；LoRA-down 回归 `torch.mm`。 |
| [PR #134：AWQ W4A16 的三级执行路径](Comfy-Kitchen-134-AWQ-W4A16-WMMA.md) | 为什么同一个 W4A16 运算需要三种实现？ | small-M 用 wave GEMV，medium-M 用 fused WMMA 复用 weight，large-M 用 dequant + library GEMM。 |
| [PR #157：Triton 中的 FP8 随机舍入](Comfy-Kitchen-157-Triton-Stochastic-FP8-Rounding.md) | 怎样匹配 CUDA 舍入语义并避免 eager 中间张量？ | 单个 Triton kernel 消费外部 RNG byte，并将 FP8 结果覆盖回 RNG storage。 |

## FlyDSL

| 笔记 | 核心问题 | 关键结论 |
| --- | --- | --- |
| [PR #1057：为 gfx1151 选择合适的 GEMM Tile](FlyDSL-1057-RDNA3-GEMM-Tile-Policy.md) | 为什么 shape-only heuristic 还不够？ | tile policy、JIT target 与 cache identity 必须使用同一个实际 device architecture。 |
| [PR #1113：Fused RoPE 如何支持 Head Dimension 96](FlyDSL-1113-Fused-RoPE-Head-Dim-96.md) | 为什么 D=96 会暴露 vector width 与 XOR pairing 的隐藏假设？ | 使用原生 `{1,2,4,8}` copy width，并为 12-lane half 改用显式 `+/- span` 配对。 |

## 阅读入口

第一次阅读可以先从 PR #127 建立“线程映射、向量读取与 fallback”的基本模型，再按兴趣选择路线：

- 量化与低比特矩阵计算：PR #129、PR #134、PR #157
- RDNA3/FlyDSL kernel 设计：PR #1057、PR #1113

性能数字均来自对应 PR 的公开记录；本文没有把 microbenchmark 外推为端到端模型收益。
