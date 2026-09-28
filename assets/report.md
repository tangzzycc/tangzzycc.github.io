# CL=4096 TMAN MHA2SHA 优化项目汇报

日期：2026-09-28

## 1. 项目背景

Qwen3-1.7B 在 SA8797P 上使用 TMAN W4A16、KV8 和 MHA2SHA 运行时，CL 从 1024 增大到 4096 后，Decode 性能明显下降。在相同模型、量化和输入条件下，Decode 吞吐由约 49.5 tok/s 降至 34.5 tok/s，降幅约 30%。

Optrace 表明，CL 增大带来的额外 HTP cycles 主要集中在长序列 Attention：MatMul 占新增 cycles 的约 47.7%，Cat 占约 28.1%，两者合计超过 75%。其中 QK 和 AV 都需要处理完整静态 CL，而原始 MHA2SHA 在 V-cache 物化、MatMul shape 和逐 query-head 调度方面还存在额外开销。相比之下，TMANLinearFused 本体随 CL 基本不变，因此本轮优化重点放在 Attention 与 KV 消费路径，而不是继续调整线性层 kernel。

本项目围绕 CL=4096、AR=1 Decode，依次从减少大张量物化、改善 HTP MatMul shape 和降低小算子调度次数三个方向优化 MHA2SHA。

## 2. 核心优化方案

### 2.1 V-cache 物化消除与 AV 分段计算

首先调整 V-cache 分支的图调度。原转换会较早生成 V-cache 的 slice/cat 节点，使这些大尺寸中间张量跨越 QK、mask 和 Softmax 阶段长期存活。优化后将它们延迟到 Softmax 之后、对应 AV MatMul 之前再生成，从而缩短 V 数据的生命周期，降低 Attention 执行期间的并发内存压力，并保持 GQA 仅存储 `Hkv` 份 V 数据，避免扩展为 `Hq` 份物理副本。

进一步利用矩阵乘法分配律，将原始计算：

```text
Output = P @ cat(Vpast, Vnew)
```

改写为：

```text
Output = Ppast @ Vpast + Pnew @ Vnew
```

该改写利用 `P @ [Vpast; Vnew] = Ppast @ Vpast + Pnew @ Vnew` 的数学等价性，删除了每层、每个 Decode token 对完整 CL=4096 V tensor 的物化。V-cache 的读取直接进入两段 AV MatMul，不再先写出一个完整拼接结果；同时保持 K/QK 路径、KV-head 共享、量化边界和 TMAN projection 不变，为后续 MatMul shape 优化建立了基础。

### 2.2 HTP 友好的 AV 对齐分块

直接按 past/new 拆分后，CL=4096 的两个 AV MatMul reduction dimension 为 `4095` 和 `1`：前者差一个元素才对齐到 4096，后者是启动成本和 tile 利用率都较差的极小 MatMul。

因此将 AV 重新划分为按 128 对齐的长前缀和小尾段：

```text
优化前：AV = MatMul(K=4095) + MatMul(K=1)
优化后：AV = MatMul(K=3968) + MatMul(K=128)
```

两种拆法的总 MAC 与 MatMul 数量基本相同，但 `3968` 和 `128` 均为 128 的整数倍，更适合 HTP/HMX kernel 的 tiling：长前缀能够使用规则 tile，尾段也不再是利用率很低的 `K=1` 退化计算。该结果说明，HTP 上 MatMul 的 shape 质量与算子数量同样重要。

端侧实测 Decode 提升约 **3 tok/s**，MatMul cycles 明显下降。优化后的目标区域 MatMul+Cat 为 45.853M cycles，而 native QNN 为 45.435M cycles，两者仅相差约 0.418M、不到 1%，表明由 V 分段引入的额外成本已基本消除。

### 2.3 GQA 双 Query Head 合并

Qwen 的 GQA 配置为 `Hq=16`、`Hkv=8`，每个 KV head 恰好由两个 query heads 共享。原实现仍为两个 query heads 分别运行 QK、mask、Softmax 和 AV；优化后将它们沿 MatMul 的 M 维合并：

```text
优化前：[1, 1, 1, 128] @ [1, 1, 128, CL]，每个 Q head 独立执行
优化后：[1, 1, 2, 128] @ [1, 1, 128, CL]，两个 Q heads 组合执行
```

QK、scale/mask、masked-softmax 和 `3968+128` 两段 AV 全程保持 `M=2`，计算完成后再恢复原 query-head 输出布局。两个 query heads 始终使用同一个 KV head，因而没有跨 head 混合，数学结果与逐 head 计算保持一致。

该方案不减少总 MAC，但显著减少小 kernel 的启动、调度和中间张量管理开销。按 28 层统计，QK 与 Softmax 从 448 个降至 224 个，AV MatMul 从 896 个降至 448 个，AV Add 从 448 个降至 224 个。端侧在 AV 对齐优化基础上继续提升约 **2 tok/s**，证明提高 query-row 批量程度能够在 HTP 上获得额外收益。

## 3. 工程实现与正确性保障

全部优化集中在 MHA2SHA 图转换阶段，不修改 TMAN custom op ABI，也不增加新的模型参数或运行命令。最终实现继续使用完整长度 QK，保留 `Hkv` 级 K/V 共享、KV8 requantize、shared-R3 权重共享以及 masked-softmax 的 `amin + where` 语义。

优化仅在能够安全识别的 AR=1 Decode 图上生效；MHA、MQA、Prefill、其他 GQA ratio 或不支持的 score 图均回退到原有转换路径。开发期测试覆盖 CL512/1024/4096、FP32/FP16、MHA/GQA/MQA、连续 cache 更新、量化 metadata 和独立/融合 TMAN projection，转换前后 CPU 数值保持一致。

## 4. 最终结果与结论

按现有端侧测试口径，AV 对齐和 GQA 合并两项后续优化累计带来约 **5 tok/s** 的 Decode 提升。整个优化过程没有依赖减少模型计算量，而是通过改善图调度、内存访问、MatMul tiling 和 kernel 粒度，提高了相同数学计算在 HTP 上的实际执行效率。

本项目表明，CL=4096 下仅关注理论 MAC 数不足以保证性能。HTP 实际效率同时受大 tensor 生命周期、完整 KV 物化、MatMul reduction 对齐以及小 kernel 数量影响。本轮优化从这三个层面逐步收敛，使 TMAN MHA2SHA 的长上下文 Attention 路径接近 native QNN 成本，并获得可复现的 Decode 吞吐提升。
