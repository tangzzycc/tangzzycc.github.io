---
title: "Expert Parallel：Token 交换、负载均衡与并行组"
---

# Expert Parallel：Token 交换、负载均衡与并行组

<div class="study-notes study-analysis" markdown="1">

来源：提供的「算能科技二面」「小米基模组一面」「阿里平头哥一面」记录；对应推理优化[第 67～71 题](../index.md#q49)。MoE 的模型结构见[模型问答](../../models/index.md)。

## 1. 专家切分与 Token 流程 {#expert-token-flow}

1. <strong class="study-key">Router。</strong> 为每个 token 选择 top-k 专家，保存 expert ID 与组合权重。
2. **Dispatch。** 将 token 发到专家所在 rank，携带返回所需的 token/slot 映射；本地路由可避免跨卡发送。
3. **专家 FFN。** 按专家聚集输入，用 grouped GEMM 等执行；专家内部也可能做 TP。
4. **Combine。** 返回各 token—expert 结果，按模型规定的权重和顺序合并，恢复原 token 布局。

这与 TP 的输出部分和归约不同：EP 交换的是路由输入和专家结果。标准 All-to-All 只是实现方式之一，专用后端可能融合打包、交换和归并。

### Dispatch 的排列和 Combine 的逆映射怎样对应？

两个 token x₀、x₁，各选 2 个专家，形成 `(0,e₁),(0,e₃),(1,e₃),(1,e₂)` 四个 token—expert 对。按专家桶化后连续输入可排列为 `[x₀@e₁,x₁@e₂,x₀@e₃,x₁@e₃]`；需要保存原 token ID、slot/路由权重及专家偏移。专家输出回来后，逆排列把每份结果送回原 token，再按各自权重相加。

跨 rank dispatch 通常先统计目的 token 数，再算发送 offset、打包和交换；动态接收数与 padding 会影响 buffer 与 Graph shape。Grouped GEMM 只执行其中的专家矩阵，不能替代路由、交换或逆映射。空专家、重复目标以及 shared expert 分支都需保持模型规定的合并语义。

### All-to-All 为什么传 Activation，不能拿 KV 量代入？

进入 FFN/MoE 的 token 隐藏表示是 `[T,H]`，路由把它送到专家位置；历史 KV 属于 Attention 状态，不是专家输入。每个 token—expert 对的输入/输出通信主载荷按 H 与通信 dtype 算。目标相同的多个专家有时可共享一次 token 发送，但需要接收端再分发，实际去重与 metadata 要按后端确认。

返回面试题：[67](../index.md#q49) · [68](../index.md#q50)
{.study-source}

## 2. 通信量如何估算？ {#expert-traffic}

设 T 为整个 EP 组进入 MoE 层的 token 数，k 为每 token 选择的专家数，H 为 hidden size，s 为通信元素字节数。忽略元数据、padding、专家内部 TP，并假设每个 token—expert 对分别传输入和输出：

- **逻辑载荷。** Dispatch 和返回各约 `T×k×H×s`。
- **远端载荷。** 仅远端专家需要跨设备传输；若远端比例为 r，二者合计发送载荷粗估 `2rTkHs`。
- **实现影响。** 同目标 token 去重、低精度通信、拓扑分层、对齐和多跳转发会改变实际流量；本式不表示每卡载荷或每条链路的字节数。

这里交换的是 **Activation 与路由元数据/专家结果**，不是 Attention 历史 KV；P/D 交接的 KV 属于另一条通信路径。Prefill 中 T 为本轮实际参与的 token 行数，Decode 中通常为本轮活动 token 总数，投机验证等路径需按实际工作另计。

例如 T=128、k=2、H=4096、BF16、远端比例 r=7/8，dispatch/返回逻辑载荷各为 2 MiB，合计远端发送粗估为 3.5 MiB。此值是全 EP 组两方向发送载荷，不是每卡、收发总和或每条物理链路流量；专家偏斜与实现去重会改变分布。

## 3. 负载不均衡在哪里产生？ {#rank-skew}

- **专家热度。** token 内容与 batch 组成让 Router 偏向部分专家；专家数相同不保证 token 数相同。
- **Rank 放置。** 多个热点专家放在同一卡上，会集中计算、容量和通信压力。
- **组内等待。** 各 rank 协同交换和推进，慢 rank 可能限制整轮；Attention 的 DP 入口也可能在专家组通信处耦合，不能始终当完全独立副本。

例如 4 个 rank 的 token—expert 分配量为 `[400,200,200,200]`，在单位工作成本相同且无通信的简化条件下，最大负载是平均值的 1.6 倍。仅看平均值会掩盖慢 rank；真实 grouped GEMM 的耗时还依赖专家形状与批量。

### 平均专家 token 数均匀，为什么一个 rank 还是慢？

每个 rank 可能有多个专家，热点聚集后总工作更大；同样 token 数对应不同专家 M、tile padding、dtype/TP 和实际设备速度，也不等价时间。Dispatch/combine 的多目标通信还可能在某链路拥塞，慢 rank 的 collective 等待会限制整个组。

先看每专家 token 分布，再看每 rank 的打包、传输、Grouped GEMM 与等待时序。只有逻辑专家热度偏斜时，重新映射/热点副本针对的是主要问题；若瓶颈是链接或本地布局，盲目迁移专家未必有益。保持模型语义的物理均衡见[专家映射与切换](11-ExpertParallel与负载均衡.md#physical-balancing)。

返回面试题：[69](../index.md#q51)
{.study-source}

## 4. 运行时均衡与模型均衡 {#physical-balancing}

- **重新放置。** 将热点与冷专家混合，依据窗口统计更新映射；权重迁移和安全切换需有开销预算。
- **冗余副本。** 多个物理副本对应同一逻辑专家，token 分流；增大显存需求，不等于修改专家函数。
- **执行优化。** 合适组批、通信计算流水线和拓扑感知路由可改善利用率；减少等待的同时检查 TTFT/TPOT。
- **语义。** 训练用损失、路由 bias 等约束专家热度；Serving 不能随意改路由选择或丢弃超量 token，除非模型本身定义了相应机制。

逻辑专家 e 可以有多个物理副本，只要它们执行相同参数/函数，路由到这些副本之一不改变 top-k 的逻辑选择。维护 `expert ID→物理候选 ranks` 映射，再按队列、拓扑和容量分流，与把 token 改送到另一逻辑专家有本质差别。

专家迁移需要先准备新权重、确保新映射可用，再切换新请求；旧位置仍需完成已经接收的 token，不能马上释放。热点检测的窗口过短容易追逐随机波动，窗口过长又跟不上流量变化；复制和迁移成本要被未来节省的等待摊薄。

返回面试题：[70](../index.md#q52)
{.study-source}

## 5. DP、TP、EP 不能盲目相乘 {#parallel-groups}

在 vLLM 文档给出的 `--enable-expert-parallel` 模式中，专家通信组覆盖一个 PP stage 内的 Attention DP×TP ranks，`EP=DP×TP`。

| 8 卡配置，PP=1 | Attention | 专家层 | 主要取舍 |
| --- | --- | --- | --- |
| DP=8、TP=1 | 8 组，Attention 权重各自复制 | EP=8，专家分布在这 8 卡 | 更多 Attention 副本，各卡处理不同入口工作 |
| DP=4、TP=2 | 4 组，各组 2 卡协作 | EP=8，复用同一批 8 卡 | Attention 每卡权重分片，但增加 TP 通信 |

EP 不是这里的第三个独立设备维度。原记录的 `DP×EP≈EP×TP` 未给出定义，不能作为配置或性能规律；应分别列出 Attention 组、专家组、专家内部 TP 与实际设备数。

资料：[vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)，查阅日期 2026-10-13；关系与可用通信后端以使用版本为准。

返回面试题：[71](../index.md#q53)
{.study-source}

</div>
