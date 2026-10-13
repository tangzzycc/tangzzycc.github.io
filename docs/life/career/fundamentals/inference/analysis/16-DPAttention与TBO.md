---
title: "DP Attention 与 TBO：缓存归属、专家通信和两批重叠"
---

# DP Attention 与 TBO：缓存归属、专家通信和两批重叠

<div class="study-notes study-analysis" markdown="1">

来源：提供的「百度昆仑芯一面」记录；对应推理优化[第 72～74 题](../index.md#q71)。DP Attention 与普通模型副本、TP/EP 的关系见 [EP 并行组](11-ExpertParallel与负载均衡.md)。

## 1. DP Attention 改变的是层级并行布局 {#attention-groups}

普通完整模型 DP 副本处理独立请求，并拥有完整逻辑模型。DP Attention 则只让 Attention 侧按请求/token 工作分组，各组保留自己负责的 KV；进入 MoE 层时可把 token 发送给跨更大范围放置的专家。

例如 8 卡：Attention 每卡处理不同请求，EP=8 的专家分布在同一批设备；Attention 并行与专家并行复用设备，而非产生 8×8 卡。也可以让一个 Attention 组内部做小 TP，此时一份请求由组内协作。

DP Attention 常与 EP 配合，但不是概念上必须有 EP。某些布局仍会为 dense FFN 或其他层做 TP/collective，必须按层描述，不能宣称 Attention 独立就代表整模型无通信。

Attention 侧保存每个请求的历史，适合按请求分组维持局部 KV；MoE 的专家很多，适合跨更大设备组分配权重。DP Attention+EP 让这两种结构采用不同并行布局：每个 Attention 组完成自己 token 的 Attention，MoE 边界再 dispatch 到专家组，combine 后返回原入口布局。

代价从“同请求多 rank 复制/协作 Attention”转向“Attention 权重副本和跨入口专家交换”。长上下文下减少重复 latent/KV 可能很有价值；低并发或偏斜请求分配时，各入口 batch 太小也可能降低矩阵效率。应同时算总 KV、权重、专家通信和真实延迟。

返回面试题：[72](../index.md#q71)
{.study-source}

## 2. TP 的 KV 冗余有哪些来源？ {#kv-replication}

| 路径 | KV 是否可正常分片 | 需注意什么 |
| --- | --- | --- |
| 普通 MHA head TP | 常按不同 KV heads 分片 | head 数可分、映射与位置一致 |
| GQA/MQA，KV heads 少于 TP | 某些实现复制 KV heads | 每卡一份不等于逻辑无重复 |
| MLA 的共享 latent，按 Query head TP | 某些路径每 rank 保存同一 latent | Query 分片不自动分片共享 latent |
| CP 等上下文分片 | 历史可沿位置分担 | 需要跨分片 Attention 状态或数据交换 |

例如 2 个 KV heads、TP=8，每卡保存 1 个 KV head 且各逻辑 head 复制到 4 个 rank 的布局中，全组保存 8 份 head 载荷，是逻辑 2 份的 4 倍。复制倍数取决于实现，不能普遍用 TP size 作为倍数。

若 MLA 每个请求 latent 在 8 rank 完整复制，改为请求由单个 Attention rank 管理可减少这类重复；同时 Attention 权重可能复制、batch 分布和专家通信变化。需比较相同全局工作与设备预算下的总权重、KV、激活和执行成本。

返回面试题：[73](../index.md#q72)
{.study-source}

## 3. TBO 怎样提供独立工作？ {#two-batch-pipeline}

TBO 将一批独立请求/token 工作拆成 A、B 两份 micro-batch，分别维护状态，交错推进。可在 A 发送专家输入或等待返回时，执行 B 已满足依赖的 Attention/GEMM；再交换角色。

| 一段时间窗口 | Micro-batch A | Micro-batch B |
| --- | --- | --- |
| 初始 | 完成当前 Attention | 准备自己的输入 |
| 重叠窗口 1 | Dispatch/等待数据 | 执行不依赖 A 的 Attention |
| 重叠窗口 2 | 专家计算 | Dispatch/等待自己的数据 |
| 后续 | Combine，再消费本批结果 | 根据本批就绪状态继续 |

这是依赖关系示例，不规定所有后端都按此时序。不得让 A 的专家计算在 A 数据未就绪时启动，也不得读取 A 尚未 combine 完成的结果。

### TBO 的重叠窗口为什么必须来自不同 micro-batch？

同一批的专家 GEMM 依赖它的 dispatch 完成，下一层 Attention 又依赖 combine 完成；把它们放到不同 stream 并不能消除依赖。另一批 B 若有独立输入和状态，就可在 A 通信期间执行 B 的 Attention，形成真正可重叠的窗口。各 micro-batch 的元数据、输出和 KV 所有权需要隔离。

拆批同时改变 M：一个大 GEMM 变成两个小 GEMM，Tensor Core 利用、专家 token 分布、padding 与 launch 次数都会变化。评估需同时用相同拆批 shape 的串行对照，区分小矩阵代价与重叠收益，再看完整批次和请求延迟。

通信可能也消耗 SM、register 和 HBM；并发时计算本身变慢，时间线看似重叠但关键路径未缩短。有效隐藏量是串行真实成本减去并发真实成本，而不是独立测试两个时间的较小值。

返回面试题：[74](../index.md#q73)
{.study-source}

## 4. TBO 的成本与适用边界 {#overlap-cost}

- <strong class="study-key">Shape。</strong> 拆半使本地 GEMM 的 M 变小，可能降低 tile 效率；也可能降低某些临时工作集，但新增/重叠缓冲未必让总峰值减半。
- **资源。** 通信 kernel 的 SM、内存和同步成本可能影响同时计算；GPU 并发能力与通信后端共同决定能否重叠。
- **调度。** 两份 micro-batch 的元数据、stream/event、完成与回收需隔离且有对应关系；CPU 过早阻塞也会破坏流水线。
- **概念。** TBO 改变批次执行组织，P/D 分离改变资源边界，PP 按层划 stage；它不减少原始通信字节，也不预测未来 token。

同一总负载比较开关，再用相同 micro-batch shape 的对照区分“重叠收益”和“小矩阵代价”。测关键路径、单 kernel 和通信时长及端到端 goodput，不能只相加独立测试的时间。

资料：[SGLang DP/DPA Guide](https://docs.sglang.io/docs/advanced_features/dp_dpa_smg_guide.md)、[SGLang Large-scale EP：DP Attention 与 TBO](https://lmsys.org/blog/2025-05-05-large-scale-ep/)。不同框架的 TBO/DBO 是相关执行思路，具体切分和调度不应仅凭名称认定完全相同；资料查阅日期 2026-10-13。

返回面试题：[74](../index.md#q73)
{.study-source}

</div>
