---
title: "DP 与 PP：服务副本、流水线和延迟"
---

# DP 与 PP：服务副本、流水线和延迟

<div class="study-notes study-analysis" markdown="1">

来源：提供的「腾讯混元 AI Infra 部门一面」「算能科技二面」记录；对应推理优化[第 48、56～58、85～90 题](../index.md#q37)。

## 1. DP 的复制范围 {#dp-replica}

- <strong class="study-key">逻辑副本。</strong> 每个 DP 副本处理独立请求或批次，持有完整逻辑模型；纯 DP 可一副本一卡，混合并行可一副本包含多个 TP/PP rank。
- **容量。** 单独增加副本不降低每副本的权重需求；模型放不下需要模型分片、量化或卸载。
- **训练与 Serving。** 经典训练 DP 同步梯度；Serving 固定权重，不需要这类每步同步。MoE、引擎协调或状态分片是另外的通信与机制。

### 一个 DP 副本包含多个 TP rank 时，完整模型到底在哪里？

“完整逻辑模型”以执行组为单位。DP=2、TP=4、PP=1 时有两个独立请求入口，每个入口通过 4 个 TP rank 完成完整模型；每卡只持有其 TP 权重分片，而不是 8 卡都各有一份全权重。单独提高 DP 复制的是执行组，不能降低每组原有容量需求。

Serving 固定权重，无需训练式每步梯度归约；但组内 TP、混合 EP 及某些引擎协调仍有通信。DP 能减少排队和增加总请求供给，不自动降低无排队单请求的层计算时间。特定 MoE 的 Attention DP 与专家共享组另有定义，见[层级并行组](16-DPAttention与TBO.md#attention-groups)。

返回面试题：[48](../index.md#q37) · [58](../index.md#q40)
{.study-source}

## 2. DP 与 TP 的资源选择 {#dp-vs-tp}

- **吞吐。** 多副本并行消费不同请求，提高可服务容量；低负载时不能凭副本数保证线性收益。
- **延迟。** DP 可能减少排队，不直接拆单请求计算；TP 影响每步计算和通信。
- **固定预算。** 例如 8 卡可比较 1 个 TP=8 副本与 2 个 TP=4 副本，前提是容量和执行约束都满足，再按 SLO 下的 goodput 选择。

固定 8 卡，假设 TP=8 的一个执行组每秒完成 10 个请求，TP=4 的每组完成 7 个请求，则两个 TP=4 副本的理想供给为 14 请求/s。但若 TP=4 单请求 TPOT 超出目标或模型放不下，这个部署不可用；反过来，低到达率时两副本也未必都能形成高效 batch。

DP 提升的是独立请求的并行供给，可能通过减少排队改善响应；不会把一个请求的每层工作拆给不同副本。Classic Serving DP 没有训练式梯度 All-Reduce，但混合 EP 的共享专家通信可能跨 Attention 入口，因此应明确“副本独立”成立在哪些层。

返回面试题：[56](../index.md#q38) · [57](../index.md#q39)
{.study-source}

## 3. PP Stage 与 Micro-Batch {#pp-stage}

- **按深度切层。** Stage 保存本地层与对应 KV，边界传激活；可在一个 stage 内继续 TP。
- **平衡成本。** 层数相同不代表耗时相同，要考虑 embedding、LM head、不同层结构、通信与内存预算。
- **Micro-batch。** 不同小批次在不同 stage 同时推进；批次切分与单请求 Prefill token 切块不同，可结合但不等同。

### PP 边界传什么，KV 又放在哪里？

Stage 本地执行自己的模型层，保存这些层的历史 KV/状态，向下一 stage 传当前 token/micro-batch 的 hidden activation 与所需身份/位置元数据。单纯按深度 PP 不要求每一步把全部历史 KV 送到下一 stage；各 stage 持有各自层状态。

Micro-batch 是一组可独立推进的工作；多份同时在不同 stage 执行，需分别跟踪 buffer、请求身份和位置。最后 stage 的采样结果还需驱动下一轮首 stage，在线 Decode 因而存在反馈链。分层的 KV、变长 batch 和结束/取消需要协调，bubble 时间表见[流水线时隙](09-DP与PipelineParallel.md#pipeline-bubble)。

返回面试题：[85](../index.md#q41) · [86](../index.md#q42)
{.study-source}

## 4. Bubble 的简化算例 {#pipeline-bubble}

只做前向，p 个均匀 stage，每个 micro-batch 在每 stage 耗时 t，m 个 micro-batch 持续供给，忽略通信和调度。

- **总时间。** 理想完成时间为 `(m+p−1)t`，p−1 表示填充与排空共同带来的额外流水线阶段。
- **资源空闲比例。** 总资源容量为 `p(m+p−1)t`，有效工作为 `pmt`，空闲比例为 `(p−1)/(m+p−1)`。
- **例子。** p=4、m=8 时，空闲比例为 3/11，约 27.3%；增加 m 可摊薄，但必须有独立工作，且小批次效率与等待需要计入。

取 p=3、m=4，假设每 stage 耗时都为 t：

| 时隙 | Stage 0 | Stage 1 | Stage 2 |
| --- | --- | --- | --- |
| 1 | A | — | — |
| 2 | B | A | — |
| 3 | C | B | A |
| 4 | D | C | B |
| 5 | — | D | C |
| 6 | — | — | D |

总时隙为 6，18 个 stage 时隙中有效工作为 12，bubble 为 1/3，与 `(p−1)/(m+p−1)` 一致。若某 stage 耗时为 2t，这张等时表不再成立，慢 stage 的供给率会控制流水线。

返回面试题：[87](../index.md#q43)
{.study-source}

## 5. 减少 Bubble 与平衡 Stage {#stage-balance}

- **供给。** 增加独立 micro-batch、平衡 stage，并重叠有独立依赖的通信；交错 stage 还会增加调度和通信复杂度。
- **不均衡。** Profile 找慢 stage，再调整层分配、stage 内 TP 或设备资源；实际划分可能在 Prefill 与 Decode 间需要取舍。

### 等层数切分为什么在 Prefill 和 Decode 都可能不均衡？

Embedding、LM head、视觉编码、MoE 和不同 Attention 层的成本不同，输入输出长度还会改变计算/访存比例。一个 stage 多放 LM head，Decode 时可能较慢；另一 stage 含长上下文 Attention，Prefill 或长 Decode 又可能成为瓶颈。平衡依据应是目标 workload 的实测时间与内存，不是层数相等。

可调整层划分、stage 内 TP 或设备分配，但每次变更还改变边界激活传输、权重/KV 容量和 kernel shape。流水线稳态供给受最慢 stage 限制；为填平 bubble 而积攒更多 micro-batch 又会增加等待，所以在线目标下最优划分未必是离线峰值吞吐最优。

返回面试题：[88](../index.md#q44) · [89](../index.md#q45)
{.study-source}

## 6. 在线 PP 的延迟与状态协调 {#serving-state}

- **延迟。** 单请求仍按 stage 顺序推进，末端采样再驱动下一步；迁移激活、缓冲、取消与各层 KV 的生命周期都需要协调，训练流水线公式不能直接套到在线 Decode。
- **服务约束。** 严格 TTFT/TPOT 目标限制为填满流水线而积攒工作；低并发下 bubble 难以摊薄。PP 能分担模型容量，跨节点时也可能避免更密集的层内归约，是否采用应按负载与互联比较，不能概括为在线 Serving 一律少用。

在线 Decode 还存在整模型的反馈环：一个 micro-batch 到最后 stage 采样后，下一 token 才能回到首 stage 开始下一步。每个 stage 必须让同一个请求的 token 位置与本地层 KV 对齐；若各 stage 的队列更新或取消进度不一致，可能产生不匹配输入或迟到写入。

单请求无通信的理想层计算总量并不因切 stage 自动增加；额外延迟主要来自边界通信、排队、批次效率和反馈调度。PP 的复杂性来自多份在途激活、分散的层状态及结束/失败协调，而不只是把层号平均分配。

返回面试题：[90](../index.md#q46)
{.study-source}

</div>
