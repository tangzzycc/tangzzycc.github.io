---
title: "Context Parallel：长上下文与跨卡 Attention"
---

# Context Parallel：长上下文与跨卡 Attention

<div class="study-notes study-analysis" markdown="1">

来源：提供的「腾讯混元 AI Infra 部门一面」「百度基模团队一面」记录；对应推理优化[第 91～93 题](../index.md#q47)。

## 1. CP 分担什么？ {#context-sharding}

- <strong class="study-key">序列维。</strong> 沿 token/context 分配激活或历史 KV，降低单卡长上下文工作集；TP 主要按特征或 head 切分，两者可组合。
- **容量与计算。** 平衡分片可分担 KV 和 Query 工作，但 dense Attention 的全局依赖仍需实现；通信和临时缓冲会增加实际占用。
- **实现边界。** Prefill 与 Decode 的 CP 可采用不同布局、通信与缓存策略；不能将按 Query 分片的计算节省直接推广到短 Query Decode。

### CP 缩小的是每卡历史，为什么仍需要全局依赖？

把历史 S 分到 p 卡后，每卡可以保存约 S/p 的相应 KV；普通全局 Attention 的 Query 仍需要所有可见位置，因此要交换 K/V tile 或合并同一 Query 的局部 max/sum/output。容量分担与正确全局输出同时要求跨卡协作，不是把序列切开后各自独立算完。

Prefill 的 Query 也可按序列分片，各卡负责不同输出行；Decode 的新 Query 很少，常需要跨历史分片汇总。因果工作量与通信重叠影响效率，长上下文容量收益要和链路、局部 kernel shape 一起看。具体合并公式见[跨分片 softmax](10-ContextParallel.md#softmax-merge)。

返回面试题：[91](../index.md#q47)
{.study-source}

## 2. Ring Attention 的执行 {#ring-attention}

- **本地对象。** 每卡保存本地 Q 与初始 KV 分片；固定 Q，沿环接收其他 KV，计算按 mask 可见的部分。
- **覆盖依赖。** 完整 dense Attention 需覆盖所有合法分片；causal mask 可以跳过部分未来位置，但要考虑不同 Query 分片的负载不均衡。
- **流水线。** 可计算当前 KV 块、传下一块，配合缓冲与同步；CP 度增加是否有益取决于计算能否覆盖通信。

两卡按序列切分 Q/K/V。以卡 0 为例，它保留本地 Query，先和本地 K/V 算部分分数，再收到卡 1 的 K/V，与同一批本地 Query 继续计算，并维护同一行的 online max/sum/output 状态。不同卡的 Query 属于不同输出行；轮转的是 K/V tile，不是把本地 Query 的输出任意平均。

Causal mask 使用全局位置：卡 0 的早期 Query 不能读取卡 1 中未来位置，即使数据已经传过来。均匀切 token 数也不一定均匀切因果 Attention 工作，早位置可见历史少、晚位置多；负载平衡需考虑实际可见配对和通信调度。

返回面试题：[92](../index.md#q48)
{.study-source}

## 3. 为什么不能平均分片输出？ {#softmax-merge}

对同一 Query 行，分片 j 保存最大分数 mⱼ、指数和 lⱼ，以及未归一化加权输出 uⱼ。合并为：

$$
m=\max_j m_j,\qquad
l=\sum_j e^{m_j-m}l_j,\qquad
u=\sum_j e^{m_j-m}u_j,\qquad O=u/l.
$$

- **归一化。** 各分片分母不同，已归一化输出需按相应全局权重合并，不能直接平均。
- **精确性。** Online softmax 保留完整可见集合的归一化关系；实际浮点顺序可能产生容差内差异，全 mask 分片还需正确处理。

设两个分片对同一 Query 得到局部最大值 m₀、m₁，局部分母 l₀、l₁，未归一化输出 u₀、u₁。取全局最大值 `m=max(m₀,m₁)`，则 `l=e^(m₀−m)l₀+e^(m₁−m)l₁`，`u=e^(m₀−m)u₀+e^(m₁−m)u₁`，最终输出 u/l。各分片的归一化基准不同，必须先转换到同一个基准。

**CP 的正确性来自全局 softmax 状态合并，而不是平均局部 Attention 输出。**
{.study-core}

Ring 可以随 K/V 到达逐块更新；也可在某些 Decode 分片路径中交换局部统计与输出状态。通信对象和代价因算法而异，不能认为所有 CP 都需要 All-Gather 完整序列。

## 4. Decode 的历史 KV 分片 {#decode-sharding}

- **短 Query。** 可以让各卡保存不同历史 KV，计算同一 Query 的局部 Attention 状态，再跨卡归并；Query 分发和后续输出布局也要计入。
- **收益条件。** 长历史可分担读取与容量；短历史可能被通信启动和归并开销抵消。
- **术语。** CP、SP 在不同系统中定义不同。Ulysses 通过 All-to-All 在 sequence/head 布局间转换，不与 Ring KV 交换等同，需先明确讨论的具体方案。

## 5. Megatron 风格 SP 与 CP 的边界 {#sp-vs-cp}

常见 Megatron 风格 Sequence Parallel 配合 TP，把 Norm、dropout 等适合逐 token 的激活工作沿 token 维分片。例如完整激活 `[T,H]` 分为 `[T/p,H]`；进入需要完整 token 行的 Column 路径前 gather，Row 后可 reduce-scatter 返回 token 分片。

这种 SP 减少相关激活的重复保存/处理，不自动将全部 Attention 历史 KV 按位置分担。CP 则直接处理长上下文分片后的 Attention 依赖；其他系统称 SP 的方案可能包含 Attention 序列并行，应先明确算子范围和通信布局。

### Megatron 风格 SP 为什么不直接解决全部历史 KV 容量？

这里 SP 指将 Norm、Dropout 等逐 token 算子的激活沿 token 行切分，与 TP 的 Linear 布局转换配合。在典型路径中，row-parallel 用 Reduce-Scatter 得到 token 分片，下一个需要全 token 输入的 column-parallel 之前再 All-Gather。主要收益是减少某些复制激活，不意味着各 rank 的 Attention 已经按历史序列独立分担。

CP 则把 Attention 可见历史沿 sequence 切开，需要交换 K/V 或合并 partial Attention 状态。TP 切层内特征，普通 head TP 可各做不同 heads 的本地 Attention。虽然都可能在某处看到 `[T/p,H]` 形状，张量的逻辑身份、持久状态和通信目的不同。其他框架也会把不同方案叫 SP，应先说清采用的定义。

返回面试题：[93](../index.md#q75)
{.study-source}

</div>
