---
title: "FlyDSL Fused RoPE：D=96 修复口述"
---

# FlyDSL Fused RoPE：D=96 修复口述

以下用于首次口述，进一步追问见 [FlyDSL PR 完整问答](FlyDSL-PR.md)。

## 1. 简历中的 Fused RoPE kernel bug 如何产生，如何修复？

### 面试回答版

这个 bug 出在 wave32、head dimension 为 96 的路径，有两个问题。旧实现按 `ceil(96/32)` 选择每 lane 三个元素，但三个 BF16 是 48 bit，底层 copy/shuffle 路径无法完整表示和重建，导致编译错误。我把向量宽度限制在原生的 1、2、4、8 中，这里选 4，使用 24 个 lane。随后两半各占 12 个 lane，旧 XOR 配对又不成立，所以对非二次幂跨度改成前半加 12、后半减 12。最后对照 Q/K 和 KV cache 验证数值，并检查原有维度的生成指令未改变。

### 详细分析

1. **Kernel 的职责。** 对 Q/K 应用 NeoX RoPE，并把旋转后的 K 与原始 V 写入 cache。NeoX 将前半元素 i 与后半元素 `i+D/2` 配对，V 不参与旋转。
2. **向量宽度错误。** 旧 width=3 的跨 lane helper 按 32-bit word 重建，`3//2=1` 只能还原两个 BF16，出现 `(2,)` 与 `(3,)` 的形状不匹配。新宽度还须整除 D 和 D/2，并满足 `D/width≤wave_size`；D=96、wave32 因此选 width=4。
3. **Lane 配对错误。** Width=4 时每半 12 lane。比如 lane 4 的伙伴应为 16，但 `4 XOR 12=8`。修复对二次幂跨度保留 XOR，其他跨度按前后半区显式加减；四个 BF16 按两个 32-bit word 交换后重建。
4. **验证范围。** 已有 PR 记录覆盖 FP16/BF16 和两种 cache layout，对照 Q、K、key/value cache；D=64/128/256 在 gfx1151 上的 ISA 保持一致。D=96 旧实现无法运行，因此这是正确性修复，没有可报告的旧版相对加速。

扩展阅读：[Fused RoPE D=96 修复笔记](../../../../gpu-kernel/flydsl/FlyDSL-1113-Fused-RoPE-Head-Dim-96.md)、[FlyDSL PR 面试问答](FlyDSL-PR.md)。
