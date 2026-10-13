---
title: "Serving 调度：Chunked Prefill、路由与延迟目标"
---

# Serving 调度：Chunked Prefill、路由与延迟目标

<div class="study-notes study-analysis" markdown="1">

来源：提供的「科大讯飞二面」「算能科技二面」记录；对应推理优化[第 11、13～14、76 题](../index.md#q25)。

## 1. Chunked Prefill 的机制 {#chunk-progress}

- <strong class="study-key">进度。</strong> 同一 prompt 分多轮计算，每轮在已完成前缀上继续推进；请求身份和上下文不因切块改变。
- **例子。** 两条 Decode 各需 1 token，一条 Prefill 还剩 18 token；每轮限制其推进 6 token，可形成三轮各 `1+1+6` 的工作，而非一轮 20 行。
- **边界。** 数字是 token 工作量，不是时间比例；完整 prompt 仍需处理，特殊双向或多模态依赖还需核对能否如此切分。

Chunking 的预算应分清两个维度：本轮新增 token 数决定 Linear 的行数，历史前缀长度决定该 chunk 的 Attention 访问。相同 256 个新 token，在短历史和 32K 历史上不是同样时间。切得更小还会增加对已有历史的重复访问及调度次数。

例如总预算 8，两条 Decode 各先占 1，则 Prefill 可以推进 6；下一轮 Scheduler 重新评估活跃请求和预算。若始终优先 Decode 而不保留任何 Prefill 机会，高 Decode 负载下新请求可能长时间无法首答，所以需要入场、时间预算或公平性政策，而不只固定一个整数 chunk size。

### Token budget 能保证一次迭代不超过 20 ms 吗？

不能。预算限制工作行数，但本地 Linear shape、历史 Attention 长度、dtype、模型层结构和缓存状态共同决定时间。同样 512 行，一个 chunk 访问 1K 历史，另一个访问 32K，Attention 成本不同；多个请求的长度分布也改变 tile 与负载。

可用已测执行模型把候选 chunk/batch 转换成预计 iteration time，再结合剩余 TTFT、Decode ITL 和 KV 空间安排；不确定时使用反馈修正和安全余量。过小 chunk 增加调用与历史访问，过大 chunk 拉长 Decode 等待，调度目标是资源与延迟约束下的工作分配。

### 双向前缀为什么不能任意逐块复用？

标准因果前缀的已完成位置不依赖后续位置，适合逐块推进；双向 Prefix-LM 中早位置可依赖前缀后位置，先算一小段后就把其 KV 当最终状态可能错误。多模态区域还可能有块级完整性需求，chunk 边界必须符合模型 attention mask 和状态语义。

返回面试题：[13](../index.md#q22)
{.study-source}

## 2. 在线 Serving 为什么受益？ {#online-fairness}

- **公平性。** 模型迭代边界才有重新调度机会；限制 Prefill 工作可缩短 Decode 等待的重迭代。
- **资源互补。** Prefill 与 Decode 一起组批可能改善利用率，但不会消除共享计算和带宽竞争。
- **取舍。** 小 chunk 增加提交和历史访问次数，可能降低 Prefill 效率；预算应结合真实执行时间与 TTFT/TPOT 目标调整。

### Decode-first 为什么可能让新请求一直等？

若持续活动 Decode 占满每轮预算，Prefill 永远没有份额，新请求 TTFT 会持续增长。需要限制活动集合、给 Prefill 保留机会、按等待时间调整顺序或拒绝过载请求；其中哪种合适取决于服务目标。无限接收只会把过载转换为无界排队。

一个 mixed batch 还包含不同长度序列，每条都要独立的位置、历史 table 和有效长度。把新 token 行拼接为大 tensor 仅是执行表示，不能让一条请求读取另一条的历史或把 padding 当输出。公平性、packed 数据正确性与 token 预算一起构成调度契约。

返回面试题：[14](../index.md#q23)
{.study-source}

## 3. 多个 Decode 实例如何路由？ {#decode-routing}

- **负载估计。** 不只看请求数，还看缓存长度、预计剩余输出、有效 batch、排队与可用 KV。预测不准时用反馈修正。
- **入场约束。** 给目标实例预留当前所需空间，并考虑增长余量；检查 P/D 交接路径与缓存局部性，避免把请求送到即将频繁抢占的实例。
- **持续执行。** 初始选择后通常保持请求与 KV 在同一实例；迁移需搬数据和状态，收益必须覆盖迁移成本。
- **长度分配。** 可结合长短请求分组，避免长 Prefill 或长历史请求集中到同一副本，但要用预计耗时和公平性反馈修正。请求路由目标是副本执行组，不能向同一 TP 组的 rank 分配不同独立批次；MoE 共享 EP 组还可能有入口协调与专家负载耦合。

### 相同请求数的两个 Decode worker 为什么负载差很多？

一个 worker 有 4 条 32K 历史的长输出请求，另一个有 4 条 1K 历史的短输出请求，活动数量一样，但 Attention 历史访问、未来驻留与 KV 增长完全不同。负载估计应包含当前 batch、历史总量、剩余输出的预测、已分配在途工作和可用容量，不只是队列长度。

请求通常固定在持有 KV 的执行组；迁移活动 Decode 需要实际 KV、采样/位置及混合层状态，可能比保持局部性更贵。初始路由可选择较空闲或缓存更合适的组，Worker 内 Scheduler 再分配本轮预算；TP ranks 共同处理同一组输入，不能各自接一批独立请求。

返回面试题：[76](../index.md#q24)
{.study-source}

## 4. 怎样平衡吞吐和延迟？ {#serving-objectives}

- **明确目标。** TTFT 约束首次响应，TPOT 约束平均生成速度，ITL 尾部约束停顿；goodput 只统计按明确规则达标的请求或 token。
- **调节参数。** Batch、chunk/token 预算、入场、P/D 配比和路由会相互影响；扩 batch 可能提高整体吞吐却拉长等待与单轮时间。
- **负载验证。** 覆盖不同输入输出长度、并发、到达率和突发；比较延迟分布及达标率，达到容量上限时用背压或拒绝避免队列无限增长。

稳定服务需要到达率低于可持续完成率。接近容量上限时，微小突发就可能让排队放大；平均 token/s 达到峰值并不代表 TTFT P99 达标。扩大 batch 能提高权重复用，也会拉长本轮时间和缓存占用，优化目标应是在指定 TTFT/TPOT 或 ITL 约束下最大化有效吞吐。

至少固定模型质量、硬件数量、输入输出长度分布、到达率与缓存条件，扫负载观察 SLO 达标率和拒绝/抢占。低负载单请求延迟、饱和离线吞吐、高负载在线 goodput 各回答不同问题。

相关笔记：[Continuous Batching](<../../../../../vllm/基础篇/Continuous Batching：动态更新执行批次.md>)、[Chunked Prefill](<../../../../../vllm/基础篇/Chunked Prefill.md>)。

返回面试题：[11](../index.md#q25)
{.study-source}

</div>
