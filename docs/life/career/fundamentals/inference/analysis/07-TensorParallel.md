---
title: "Tensor Parallel：Linear、Attention 与 FFN 切分"
---

# Tensor Parallel：Linear、Attention 与 FFN 切分

<div class="study-notes study-analysis" markdown="1">

来源：提供的「腾讯混元 AI Infra 部门一面」「百度昆仑芯一面」记录；对应推理优化[第 49～55 题](../index.md#q26)。

## 1. Column 与 Row 的逻辑维度 {#column-row}

约定 `Y[M,N]=X[M,K]W[K,N]`，p 为 TP rank 数，以下假设维度可整除。

| 方式 | 每卡权重 | 每卡结果 | 常见通信 |
| --- | --- | --- | --- |
| Column Parallel | `[K,N/p]` | `[M,N/p]` 输出分片 | 下游需要完整输出时 All-Gather，否则保留分片 |
| Row Parallel | `[K/p,N]` | `[M,N]` 部分和 | All-Reduce 得到完整结果，或 Reduce-Scatter 得到归约分片 |

PyTorch Linear 常存 `W[N,K]`，所以物理行列名字会反过来；切分仍应按逻辑输入和输出特征解释。Row 路径 bias 通常在归约后加一次，避免重复累加。

完整 hidden 激活的 Norm、逐元素残差等常在常规 TP 路径中复制执行；下游布局允许时，激活也可以保留分片。Replicated Linear 可用于较小且需要完整结果的投影等路径，但并非所有 Router、Norm 或 embedding 都必须复制，应明确具体输入输出布局与代价。

Row Parallel 的输出相加来自矩阵乘法本身：把 `X=[X₀,…,Xₚ₋₁]` 沿输入特征切开，W 按同一 K 边界切成纵向块，则 `Y=ΣᵣXᵣWᵣ`。每卡的 Yᵣ 有完整 N 列，却只含部分 K 的贡献；All-Reduce 得到所有卡完整 Y，Reduce-Scatter 则在求和后只保留指定输出分片。

Column Parallel 满足 `Y=[XW₀,…,XWₚ₋₁]`，各卡得到不同 N 列，彼此不是部分和。需要完整 Y 时用拼接式 All-Gather，不能相加。<strong class="study-key">通信类型由“输出是不同特征，还是同一输出的部分贡献”决定。</strong>
{.study-core}

返回面试题：[49](../index.md#q26) · [50](../index.md#q27)
{.study-source}

## 2. QKV 与 Attention Heads {#head-sharding}

- **Head 分片。** QKV 的输出通道按对应 head 分配，各 rank 本地执行自己的 Attention；不需要为常规按 head TP 交换完整历史序列。
- **输出投影。** 本地 head 输出组成输入分片，输出投影用 Row Parallel 产生 H 维部分和，再归约；Attention mask 和位置语义保持一致。
- **GQA/MQA。** Query 与 KV head 数不同；KV heads 小于 TP 或不能匹配分组时可能复制或采用特殊映射，因此 KV 容量不能总除以 p。

### GQA 的 Q/KV 切分怎样保持对应关系？

设 h_Q=32、h_KV=8，每 4 个 Query heads 使用同一个 KV head。TP=8 时，每 rank 持有 4 Q heads 和对应 1 KV head即可本地 Attention；若按融合 QKV 的物理数组盲目切成连续 1/8，不检查 Q/K/V 分段和 head 分组，可能让 Query 读错 Key。

当 h_KV=2、TP=8 时，每逻辑 KV head 服务 16 个 Query heads，常见 head TP 路径可能在 4 个 rank 复制该 head。每 rank 只有 4 Q heads，但全组 KV 物理载荷是逻辑 KV 的 4 倍。复制源于所选 head 映射，不是所有 TP 都必然复制；CP 或其他后端布局可改变方案，但增加跨卡协作。

Attention 输出投影接收本 rank 的 head 输出作为输入特征分片，然后 row-parallel 归约成 hidden 部分和。QKV 权重的输出切分与 O 权重的输入切分配合，才避免在两者之间 gather 全部 heads。

返回面试题：[51](../index.md#q28) · [49](../index.md#q26)
{.study-source}

## 3. FFN 的连续切分 {#ffn-pair}

- **两层配对。** `H→F` 按 F 切成 `F/p`，激活本地执行；`F→H` 按输入分片得到 H 维部分和，最后归约。
- **SwiGLU。** Gate/up 必须持有对应的中间通道，才能本地执行逐元素乘法；不必先 gather 全部 F。
- **收益边界。** Column 接 Row 避免中间布局恢复；残差、Norm 或其他算子需要完整或其他布局时，通信安排可能改变。

对 SwiGLU，令 `Z=SiLU(XW_gate)⊙(XW_up)`，`Y=ZW_down`。把 F 通道切成 p 份，gate/up 在每卡保留同一组通道；SiLU 与逐元素乘法不跨通道，所以 Zᵣ 可完全本地生成。Down 按这些输入通道切分，最后 `Y=Σᵣ ZᵣW_down,ᵣ`。

这个配对避免把中间 F 维激活恢复到完整形状，但前提是中间算子可在分片内独立执行。若换成跨 F 通道的归一化或其他耦合算子，就需要额外统计/通信；不能把 Column+Row 的无中间 gather 结论推广到任意网络。

返回面试题：[52](../index.md#q29) · [53](../index.md#q30)
{.study-source}

## 4. TP 扩大后的取舍 {#tp-tradeoff}

- **资源。** 每卡权重和部分计算降低，但本地矩阵变窄，tile 效率和 head/量化分组对齐可能受影响。
- **通信。** 常规非 sequence-parallel 的 Transformer 路径，Attention 输出投影和 FFN 收缩投影通常各需一次归约；其他模型和配置另算。
- **选择。** 比较模型容量、阶段延迟、通信和固定资源下的吞吐；小 batch 或跨节点场景可能更早遇到扩展瓶颈。

### TP 为什么不总是把计算和延迟都除以 p？

粗略单层时间可写为 `T_local(shape/p)+T_collective+T_layout+T_wait`，而不是 `T_one/p`。本地矩阵变窄后，tile 尾部、指令对齐和任务数量会变化；batch 很小时启动与归约延迟不随权重减少而消失。为小 batch 使用更多 TP 还可能复制 GQA/MLA 状态。

跨多层每次投影归约都在下一层依赖链上，若没有独立工作覆盖，通信成本持续相加。固定 8 卡，TP=8 一个组和 TP=4 两个副本既改变单请求延迟，也改变组批与队列；先检查容量和单请求 SLO，再比较在线 goodput，不能只比峰值总算力。

返回面试题：[54](../index.md#q31)
{.study-source}

## 5. TP=8 的 Prefill 维度算例 {#tp8-shapes}

固定普通 head TP、无 SP/CP、T=128 个本轮 token 行、H=4096、32 个 Q heads、8 个 KV heads、head_dim=128，FFN 中间维 F=14336；逻辑 Linear 用 `[输入,输出]` 权重布局。

| 环节 | 全局逻辑形状 | 每 rank 形状或结果 |
| --- | --- | --- |
| 输入 | `[128,4096]` | 常规路径完整复制 `[128,4096]` |
| Q 权重/输出 | `[4096,4096]` / `[128,4096]` | `[4096,512]` / `[128,512]`，4 Q heads |
| K/V 各自权重/输出 | `[4096,1024]` / `[128,1024]` | `[4096,128]` / `[128,128]`，1 KV head |
| 融合 QKV 输出 | `[128,6144]` | `[128,768]`，三段按 head 映射切分 |
| 本地 Attention | Q 对完整可见历史 | 输出 `[128,512]`，本地 4 Q heads 使用对应 KV |
| 输出投影 Row | `[4096,4096]` 权重 | 本地权重 `[512,4096]`，输出 `[128,4096]` 部分和 |
| FFN gate/up 各自 Column | `[4096,14336]` | `[4096,1792]`，输出 `[128,1792]` |
| FFN down Row | `[14336,4096]` | `[1792,4096]`，输出 `[128,4096]` 部分和 |

若 Row 部分和归约使用 BF16 通信，单次逻辑载荷为 `128×4096×2=1 MiB`。Ring All-Reduce 在 TP=8 下每 rank 发送和接收各 `2×7/8×1=1.75 MiB`，调用次数与实际通信 dtype 另算。

设有效历史总长度 L，单层本地 K/V 主载荷在上述无复制布局中约 `2×L×1×128×元素字节数`；L 不是本轮 T，且多请求应取缓存 token 总数。GQA head 数减少、MLA、CP 或复制布局改变后不能照用。

Dense FFN 常用 Column+Row 配对；MoE 若用专家内部 TP，可按每专家对应中间维切分，若用 EP 则先按 token—expert 路由，不能直接把所有专家数量与每专家维度都除以 TP。

相关笔记：[QKV 与 Attention 投影](../../../../../vllm/基础篇/Attention的计算机理与vLLM实现.md)。

返回面试题：[55](../index.md#q74)
{.study-source}

</div>
