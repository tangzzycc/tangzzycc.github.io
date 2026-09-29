# CL=4096 TMAN MHA2SHA 优化项目汇报

日期：2026-09-29

## 1. 项目背景

Qwen3-1.7B 在 SA8797P 上使用 TMAN W4A16、KV8 和 MHA2SHA 运行时，CL 从 1024 增大到 4096 后，Decode 性能明显下降。Profile 表明主要增量来自长序列 Attention 的 MatMul 与 KV Cat 路径：QK/AV 需要处理完整静态 CL，原始图在 V-cache 物化、MatMul shape 和逐 query-head 调度方面仍有额外开销。

本项目围绕 CL=4096、AR=1 Decode，从减少大张量物化、改善 HTP MatMul shape 和降低小算子数量三个方向优化 MHA2SHA。

## 2. 核心优化

### 2.1 V-cache 物化消除与 AV 分段计算

将 V-cache 分支的 slice/cat 节点延迟到 Softmax 之后、AV MatMul 之前生成，缩短大尺寸 V 中间张量的生命周期，并保持 GQA 仅使用 `Hkv` 份 V 数据。

进一步利用矩阵乘法分配律，将：

```text
Output = P @ cat(Vpast, Vnew)
```

改写为：

```text
Output = Ppast @ Vpast + Pnew @ Vnew
```

由此删除每层、每个 Decode token 对完整 CL=4096 V tensor 的物化，同时保持 K/QK 路径、KV-head 共享和量化边界不变。

### 2.2 HTP 友好的 AV 对齐分块

按 past/new 直接拆分时，CL=4096 的两个 AV MatMul reduction dimension 为 `4095` 和 `1`，不利于 HTP kernel 的 tiling。优化后调整为：

```text
优化前：AV = MatMul(K=4095) + MatMul(K=1)
优化后：AV = MatMul(K=3968) + MatMul(K=128)
```

两种拆法的总 MAC 和 MatMul 数量基本相同，但 `3968` 与 `128` 均按 128 对齐，避免了 `K=1` 的退化 MatMul。端侧实测 Decode 提升约 **3 tok/s**，MatMul cycles 明显下降；目标区域 MatMul+Cat 为 45.853M cycles，与 native QNN 的 45.435M 相差不到 1%。

### 2.3 GQA 双 Query Head 合并

Qwen 的 GQA 配置为 `Hq=16`、`Hkv=8`，每个 KV head 由两个 query heads 共享。优化后将两个 query heads 沿 MatMul 的 M 维合并：

```text
优化前：[1, 1, 1, 128] @ [1, 1, 128, CL]
优化后：[1, 1, 2, 128] @ [1, 1, 128, CL]
```

QK、masked-softmax 和 `3968+128` 两段 AV 全程保持 `M=2`，计算完成后再恢复原输出布局。总 MAC 基本不变，但 QK、Softmax、AV MatMul 和 Add 的实例数减半，减少了小 kernel 的启动与调度开销。端侧在 AV 对齐优化基础上继续提升约 **2 tok/s**。

## 3. 最终结果

按现有端侧测试口径，AV 对齐与 GQA 合并累计带来约 **5 tok/s** 的 Decode 提升。最终实现保留完整长度 QK、`Hkv` 级 KV 共享、KV8 requantize、shared-R3 和 masked-softmax 语义，并对 MHA、MQA、Prefill 及其他不满足优化条件的图安全回退。

本项目表明，CL=4096 下的 HTP 性能不仅取决于理论 MAC 数，还受到大 tensor 物化、MatMul reduction 对齐和小 kernel 数量的显著影响。针对这些因素优化后，TMAN MHA2SHA 的长上下文 Attention 路径已接近 native QNN 成本，并获得了可复现的 Decode 吞吐提升。
