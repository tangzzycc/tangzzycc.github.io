# Comfy-Kitchen PR #127

> [https://github.com/Comfy-Org/comfy-kitchen/pull/127](https://github.com/Comfy-Org/comfy-kitchen/pull/127)

## PR 动机

这项 PR 解决的不是 INT8 GEMM 本身，而是 GEMM 前面的 activation quantization 已经成为瓶颈。 在 Anima05 的代表 shape `M=8192, K=2048, BF16` 中，旧 rowwise quantization 约为 `2.3 ms`，后续 INT8 GEMM 约为 `0.74 ms`。如果只优化 GEMM，端到端收益会被前处理吞掉。

旧 HIP kernel 的数学没有问题，问题在执行方式：每次循环只加载一个标量，而且 `load_in(..., in_code)` 在热循环里根据运行时 dtype code 解释 FP32、FP16 或 BF16。 PR #127 把 dtype 变成模板参数，并让每个线程一次读取 16 bytes。它是一项 “减少 load/loop/address-control 指令”的优化，不是量化算法变更，也不是把必要数据量减少了四倍或八倍。

## 