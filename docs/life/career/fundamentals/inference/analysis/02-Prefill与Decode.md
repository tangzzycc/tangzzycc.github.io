---
title: "Prefill 与 Decode：计算、带宽和利用率"
---

# Prefill 与 Decode：计算、带宽和利用率

<div class="study-notes study-analysis" markdown="1">

来源：提供的「小鹏汽车一面」记录，提取两个推理阶段的性能特征。

「科大讯飞二面」再次提及相同计算特征，沿用已有[第 3～6 题](../index.md#q4)，并补充[第 7～9 题](../index.md#q11)的指标与优化方向。

## 1. 阶段与指标 {#phase-semantics}

- <strong class="study-key">Prefill。</strong> 已知 prompt token 可一起计算，或分块完成；填充 KV 后，完整 prompt 的最后位置通常用于采样首个生成 token。
- **Decode。** 新 token 再作为下一轮输入，产生它的 KV，并采样下一个 token；历史 KV 可以复用，前向仍要执行各层。
- **指标对应。** 首 token 延迟包含排队、预处理和 Prefill 等成本；生成间隔主要反映 Decode 及其等待。不能把两者都归为单个 kernel 耗时。

设 prompt 为 `[A,B,C]`。Prefill 在因果 mask 下同时计算三个位置；C 的输出 logits 用于采样 D。下一轮 Decode 输入 D，生成 D 的各层 K/V，再根据 D 的最后层输出采样 E。因此“已输出 D”与“缓存已包含 D”之间可能差一轮前向，Scheduler 的已计算长度必须和输出长度分开记账。

Prefill 可并行是因为输入 token 已知，位置间的依赖由 mask 定义，而不是因为它没有因果关系。对于 Prefix-LM 的双向前缀，或有额外视觉/状态依赖的模型，是否能按任意 token 边界切 chunk，需要检查既定模型语义。

返回面试题：[03](../index.md#q4)
{.study-source}

## 2. 为什么 Prefill 更偏计算？ {#prefill-intensity}

- **统一 shape。** Linear 写成 `X[M,K] × W[K,N] → Y[M,N]`，M 是本轮 token 行数，运算量约为 `2MNK`。
- **权重复用。** 当权重读取主导、每个权重元素为 s 字节且忽略其他流量时，算术强度约为 `2M/s` FLOP/byte。BF16 下 M=1 约为 1，M=32 约为 32。
- **边界。** Prefill 的 M 通常较大，更容易提高复用与矩阵效率；真实流量还含激活、输出、重复读取和 Attention，不能直接用上述简化值预测延迟。

使用同一 `X[M,K]W[K,N]`，若所有输入输出各读写一次，元素均为 s bytes，理想主流量为 `s(MK+KN+MN)`，强度为 `2MNK/[s(MK+KN+MN)]`。当 M 较小且 KN 最大时，退化为前述 `2M/s`；M 继续增大后，激活和输出也显著，不能无限使用只读权重的近似。

例如 K=N=4096、BF16：M=1 时计算约 33.6 MFLOPs，主权重约 32 MiB，强度接近 1 FLOP/byte；M=128 时计算约 4.29 GFLOPs，连同输入输出的理想流量约 34 MiB，强度约 120.5。这个增长解释了 Prefill 更容易进入计算主导，但实际 tile 和 Attention 流量仍需计入。

### Prefill 的 Attention 成本为什么和 Linear 不同？

设 B 个请求、每个长度 S、Query heads h、head width d。QK 与 PV 的 dense Attention 主计算约为 `4BhS²d` FLOPs，因果路径只需三角可见配对，但实际 tile 对角、mask 和实现会影响工作量；Linear 通常随 token 行数 BS 线性增长。长 S 时 Attention 的增长更快，不能只用权重读取强度解释全部 Prefill。

Chunked Prefill 的新行数 C 决定 Linear 的 M；该 chunk 对已计算历史 P 和本块可见前缀做 Attention，配对规模近似 `CP+C(C+1)/2`，不是每个 chunk 都只算 C²。切块减少每次独占时间，但已有历史访问仍在，chunk 大小与剩余历史必须共同进入性能模型。

返回面试题：[04](../index.md#q5)
{.study-source}

## 3. 为什么小 batch Decode 更偏带宽？ {#decode-bandwidth}

- **少量计算也要读权重。** 每轮 token 行数少，权重流量难以摊薄；权重不能常驻片上缓存时，数据供给成为明显成本。
- **KV 也要读取。** 上下文越长，Attention 的历史访问通常越多；KV head 数、缓存精度与模型架构决定实际载荷。
- **优化取舍。** 权重量化降低权重流量，合批提高复用；但合批也可能增加等待，量化还需计入反量化与精度成本。

一个包含 P 个参数的 dense 模型，粗略权重前向工作量约 2P FLOPs/token；batch 中 M 个 token 在同次算子中复用权重时，工作量近似 2MP，而权重读取不必变为 M 倍。小 M 没有足够复用，量化主要降低这一项；较大 M 改善强度，却可能让每次迭代更长。

Decode Attention 的另一项是历史 K/V：每步读取载荷随活动请求各自的上下文长度增长。长历史时，即使权重已经 INT4，BF16/FP8 KV 仍可能成为主要流量。GQA 复用、KV 量化和 Split-KV 针对这一段；它们不能用权重量化的收益替代。

### GQA 的理论 KV 减少，怎样变成真实读取减少？

若 h 个 Query heads 映射到 g 个 KV heads，历史 KV 的理想载荷按 g 计算，但一个简单 kernel 为每个 Query head 独立读取同一 KV 可能重复供数。后端可按组协作复用 KV，或依靠 cache 命中；真实 DRAM、L2 和 SM 读取比例因此不相同。

在 Decode 中增加 batch 可让同层权重被更多 token 复用，却无法让不同请求的不同 KV 自动变成同一份读取。历史 Attention 总工作近似取各请求长度之和，长短请求分布改变每轮耗时；同样 batch=32，短历史与 32K 历史不是同一 workload。

返回面试题：[05](../index.md#q6)
{.study-source}

## 4. 高峰值算力为何不等于高 Decode 性能？ {#peak-vs-effective}

- **带宽上界。** 简化 Roofline 为 `有效算力 ≤ min(计算峰值, 带宽 × 算术强度)`；低算术强度时，增加计算峰值不改变带宽侧限制。
- **形状与并行度。** 小矩阵可能无法高效使用矩阵指令，也可能没有足够 workgroup；算力低利用不等于一定已跑满带宽。
- **依赖与空隙。** 同一请求有自回归依赖，短 kernel 还可能受 CPU 提交影响；看设备时间线区分“等待数据”和“没有工作”。

### 带宽没满、算力也没满时，为什么仍不能靠换更强算力解决？

小矩阵可能只有少数 CTA，部分 SM 没有工作；warp 大量等同一依赖，也可能没有足够 independent load 去填满带宽。短 kernel 间 CPU 准备空隙又会让全局利用率低。这些分别是任务量、依赖与提交问题，不能都叫 Memory Bound。

先看执行时间线是否连续，再看 grid/local shape、eligible warp、流量和指令路径。为小 M 增加 Split-K 是任务量候选，Graph 是提交候选，预取/ILP 是依赖候选；哪项有效必须对应实际瓶颈，且计入额外归约、资源和状态成本。

相关笔记：[GEMM 的 Roofline 分析](../../../../../ai-infra/CUDA矩阵乘法优化：从Naive到cuBLAS.md)。

返回面试题：[06](../index.md#q7)
{.study-source}

## 5. TTFT 与 TPOT 的计量 {#latency-metrics}

- **时间戳。** 请求在 t₀ 提交，第一个和最后一个 token 分别在 t₁、tₙ 收到；常见口径为 `TTFT=t₁−t₀`、`TPOT=(tₙ−t₁)/(N−1)`，N>1。
- **边界。** TTFT 包括等待与返回，TPOT 包含生成阶段间隙；客户端文本块可能含多个 token，不能直接把流块间隔等同于内部 Decode step。
- **尾部体验。** 平均 TPOT 可能掩盖停顿，需同时看 ITL 的高百分位和请求达标率；单 token 输出不能按上述 TPOT 公式除法。

例如请求在 0 ms 提交，100 ms 收到首 token，220 ms 收到第 5 个 token，则 TTFT=100 ms，TPOT=(220−100)/4=30 ms/token。若四次内部间隔分别为 10、10、90、10 ms，平均仍为 30，但用户会经历一次明显停顿；这说明 ITL 分布提供了平均 TPOT 没有表达的信息。

**TTFT、TPOT、ITL 和吞吐必须声明测量边界及 token 计数。**
{.study-core}

TTFT 可拆为入场前排队、预处理、Prefill、采样和返回，P/D 的恢复/传输是否位于首 token 前取决于实现。吞吐还需区分请求/s、输入 token/s、输出 token/s、总 token/s；按 SLO 达标的 goodput 应说明达标规则，不能拿一个指标替代全部体验。

返回面试题：[07](../index.md#q11)
{.study-source}

## 6. 优化落在什么阶段？ {#optimization-targets}

- **TTFT。** 排队主导时改善入场、路由与资源供给；重复前缀利用有效缓存；计算主导时优化 Prefill Attention/GEMM。量化是否改善 Prefill 要看具体路径。
- **TPOT。** 优化权重和 KV 流量、Decode kernel、提交与算子间访存；控制 Prefill 干扰，避免过大的实际 Decode batch 拉长迭代。
- **共同代价。** 缓存、合批、chunking 和 P/D 会同时改变多个阶段，必须检查完整延迟曲线及相同资源下的吞吐。

### 如何从延迟拆分选择下一步优化？

TTFT 若主要是队列，单 kernel 即使更快也可能被负载分配和入场积压掩盖；若重复长前缀的 Prefill 主导，prefix reuse 或恢复才对症；若 CPU 提交空隙主导，Graph/异步准备值得比较。TPOT 则需要看每轮权重、KV、通信、调度和 Prefill 干扰，而不是只比较最后一个 Attention kernel。

每项优化会改变其他成本：缓存恢复增加 IO，量化增加解码与误差，合批增加复用又拉长 iteration，P/D 隔离增加交接。应以同资源和相同 workload 检查完整 TTFT/ITL/TPOT 分布及 goodput，保留解释收益来源的时间线证据。

返回面试题：[08](../index.md#q12) · [09](../index.md#q13)
{.study-source}

</div>
