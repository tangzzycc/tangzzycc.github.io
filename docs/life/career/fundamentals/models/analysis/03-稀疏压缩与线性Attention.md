---
title: "稀疏、压缩与线性 Attention：DSA、CSA、HCA、KDA"
---

# 稀疏、压缩与线性 Attention：DSA、CSA、HCA、KDA

<div class="study-notes study-analysis" markdown="1">

来源：提供的「算能科技二面」记录；对应模型结构[第 2、5～6、12～13 题](../index.md#q7)。模型术语根据官方技术报告与模型卡核对，查阅日期 2026-10-13。

## 1. 先判断改变了哪个维度 {#compression-axis}

| 机制 | 保存什么历史 | 当前 Query 访问什么 | 主要收益与边界 |
| --- | --- | --- | --- |
| GQA/MLA | 每位置较小的 KV/latent 表示 | 普通 dense 路径仍访问所有可见位置 | 减少单位置载荷，不自动减少位置数 |
| DSA | 候选 KV 与索引所需状态 | Indexer 选择的 top-k 位置 | 降低主 Attention 访问量；候选缓存仍可能线性增长 |
| CSA | 序列压缩 KV、索引及局部窗口 | 压缩条目的 top-k 与局部窗口 | 同时减少条目数和主 Attention 访问量 |
| HCA | 更强压缩的 KV 与局部窗口 | 全部可见压缩条目与局部窗口 | 对压缩集合做 dense Attention，仍需保存增长的压缩缓存 |
| KDA | 固定形状递归状态及局部算子状态 | 状态读取与更新 | 单层 Decode 历史状态不逐 token 增长，有限状态承载有容量约束 |

这些是架构维度的比较。FlashAttention 的执行分块与 online softmax 保持既定 dense Attention 语义；稀疏选取、压缩或递归状态通常需要模型设计与训练配合。

返回面试题：[02](../index.md#q7)
{.study-source}

## 2. DSA：先索引，再访问主 KV {#sparse-indexer}

DeepSeek-V3.2 的 Lightning Indexer 生成较轻的匹配表示，为可见历史位置评分并选择 top-k，主 MLA 再访问选中位置。Indexer 的评分、top-k 和主缓存读取都是实际成本。

例如可见历史有 32768 个位置，主 Attention 选 2048 个：主访问集合可缩小，但候选缓存和索引评分并不因此自动只剩 2048。不能用 top-k 大小代替整个系统的存储或运算复杂度。

### 先做 Indexer，为什么总成本仍可能随历史线性增长？

设历史 S，Indexer 匹配宽度 d_I，主 Attention 宽度 d_A，选 k 个位置。一种简化扫描式路径的成本包含约 `O(Sd_I)` 的评分、top-k 选择和 `O(kd_A)` 的主历史访问；没有把全部成本变成 O(k)。Indexer 缓存、候选主 KV 与选择 metadata 也可能仍按 S 增长。

收益条件是较轻评分加选择、稀疏 gather 和主计算的总成本，小于 dense 主 Attention 的成本。历史短、k 接近 S，或 gather 很不规则时，选择开销可能占主导。k 固定并不能保证所有长度下加速，也不能用 k/S 直接当压缩率。

模型训练需要让轻量 Indexer 选出对主 Attention 有价值的位置；推理不能任意用一个近似分数替换训练约定。选择集合还必须因果有效、mask 正确，索引表与物理 KV 页映射一致。

返回面试题：[12](../index.md#q8)
{.study-source}

## 3. CSA/HCA：压缩的是序列条目 {#sequence-compression}

DeepSeek V4 技术报告定义：

- <strong class="study-key">CSA，Compressed Sparse Attention。</strong> 先学习式聚合历史、沿序列减少 KV 条目，再在压缩集合使用 DSA；实际压缩可有重叠，不是简单对每组 token 求均值。
- **HCA，Heavily Compressed Attention。** 使用更高序列压缩率，对所有可见压缩条目执行 dense Attention，不使用同一 top-k 稀疏选择。
- **局部窗口。** 两者都补充未压缩的近期滑窗 KV，保留细粒度局部依赖；访问需保持因果性，未完成压缩块不能泄露未来信息。

设历史长度 L，压缩率 c，窗口 w，CSA 选择 k 个压缩条目。理想条目计数中，缓存量级约 `L/c+w`；CSA 主访问约 `k+w`，HCA 约 `L/c+w`。各层 c 不同，CSA 还需索引状态与评分工作；这些计数不等于完整 FLOPs 或真实字节数。

MLA 减少每位置的特征载荷，CSA/HCA 减少序列条目数量，讨论压缩时需要明确是哪一维。

### 序列压缩条目什么时候可以成为可读历史？

假设每 c 个历史 token 形成一个压缩条目，当前位置只能使用已经完成且符合因果语义的条目；尚未收齐的一组不能借用未来 token 完成聚合。局部窗口保留近期原始 KV，弥补压缩条目的细节损失；真实学习式压缩可能重叠，不能只按均值池化解释。

对 CSA，可见集合是局部历史加选出的压缩条目；HCA 对更强压缩的全部可见条目计算主 Attention。Cache 容量、主访问量和 Indexer 工作是三个独立计数，compression ratio 只描述其中一部分。跨层压缩率和状态不同，P/D 或 prefix reuse 需交接原始窗口、压缩 KV 和必要索引/未完成聚合状态。

返回面试题：[13](../index.md#q9)
{.study-source}

## 4. KDA：用纠错式递归状态累计历史 {#delta-state}

KDA（Kimi Delta Attention）扩展 Gated DeltaNet：

1. **遗忘。** 用通道级门对旧状态的不同特征施加衰减，而非只给整个 head 一个统一遗忘系数。
2. **纠错写入。** 利用当前 Key 在状态中的关联预测与当前 Value 的差异，进行 Delta Rule 更新，改善对新关联的写入。
3. **读取。** 当前 Query 从更新后的状态读取输出；Decode 不再逐位置扫描完整历史 KV。

以每 head 的简化状态形状 `[dₖ,dᵥ]` 看，固定模型与 batch 时状态尺寸不随历史 token 数增长，但会随 batch、head、层数和 dtype 增长。Prefill 可用 chunkwise 并行算法，不能因其递归定义就认定只能逐 token 串行执行。

用最小 Delta Rule 例子理解“纠错”，不把它当成 KDA 的完整实现公式：令状态 `S[dₖ,dᵥ]` 保存 Key→Value 关联，当前 key 为列向量 k，value 为列向量 v。旧状态的关联预测为 `Sᵀk`，写入残差为 `v−Sᵀk`，简化更新 `S'=S+βk(v−Sᵀk)ᵀ`。如果 k 是单位向量、β=1，新状态在该 key 方向的预测正好更新为 v；普通累加 `S+kvᵀ` 则会把旧预测一起叠加。

例如 dₖ=dᵥ=1、k=1、S=2、v=5，纠错后 S'=2+(5−2)=5，而简单累加为 7。KDA 在这个纠错规则上增加通道遗忘门；下面用报告中的状态式说明遗忘与写入的先后关系。

!!! tip "固定状态压缩了逐位置历史"
    单层 Decode 保存有限形状的关联状态，而不是逐 token KV 列表。它减少历史扫描，但有限状态也有表达与记忆约束；不能把这种结构当成任意 softmax Attention 的无损替换。

### Delta Rule 能否从一个优化目标推导出来？

令状态 `S[d_k,d_v]` 表示 Key→Value 关联，当前列向量 k/v 的重构目标为 `L(S)=½‖Sᵀk−v‖²`。对 S 的梯度是 `k(Sᵀk−v)ᵀ`，一步学习率 β 的梯度下降得到 `S'=S+βk(v−Sᵀk)ᵀ`。所谓“纠错”是写入目标与旧预测的残差，而不是把每份 Value 不加区分地累计。

Kimi Linear 报告 §3 的 KDA 状态式在同一列向量记号下为：

$$
S_t=(I-\beta_t k_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}
      +\beta_t k_tv_t^\top,\qquad o_t=S_t^\top q_t.
$$

α_t 是 d_k 个通道的遗忘门，β_t 控制写入，k_t/q_t 为 d_k 维，v_t/o_t 为 d_v 维。先将旧状态衰减，再写入纠错残差：

$$
\begin{aligned}
\bar S_t&=\operatorname{Diag}(\alpha_t)S_{t-1},\\
S_t&=\bar S_t+\beta_t k_t(v_t-\bar S_t^\top k_t)^\top.
\end{aligned}
$$

**残差使用的是遗忘后的状态预测。** 不同轴向或转置实现的存储式要保持等价，不能交换对角门与 rank-1 更新的顺序。

### 固定状态为何有容量限制，又怎样并行 Prefill？

不同 Key 不正交时，一次 rank-1 写入也会改变其他 Key 方向的读出，形成关联干扰；遗忘门控制旧关联保留，有限矩阵无法无损保存任意长逐位置历史。它是训练得到的状态模型，不是普通 dense Softmax KV 的无损编码。

递推可写为 `S_t=A_tS_{t−1}+B_t`，两个变换的合成是 `(A₂A₁,A₂B₁+B₂)`，具有可结合的代数形式，因此可组织 chunkwise/scan 式并行。通用 dense A 的运算昂贵，KDA 利用对角门与 rank-1 结构设计高效分块算法；“可结合”不代表直接用普通 scan 就得到最快 kernel。

Decode 的状态更新/读取主要约 `O(d_kd_v)` 每 head，不随历史 S 增长；Prefill 仍需处理全部 token，并包含局部投影、门和分块计算。资料：[Kimi Linear §2–3](https://arxiv.org/html/2510.26692v1#S3)。

返回面试题：[05](../index.md#q10)
{.study-source}

## 5. 混合模型仍有多种缓存 {#hybrid-state}

发布的 Kimi Linear 以 3 个 KDA 层对 1 个全局 MLA 层交替，利用递归状态效率与全局历史访问能力。服务框架因此要同时管理 KDA 状态、MLA KV 及必要的局部算子状态；整模型历史缓存仍含随 L 增长的部分。

状态复制、prefix reuse、取消、抢占与恢复都要按层类型处理，不能只搬一份标准 KV 张量就认为全部状态已经交接。3:1 是所述模型的配置，不是所有混合 Attention 的定律。

假设每 4 层有 3 层固定状态、1 层 MLA。固定状态部分在给定 batch 下不随 S 增长，MLA 部分仍近似按 S 增长，因此整体可以写成 `固定层状态+线性增长的MLA KV+其他局部状态`。3:1 配置降低增长系数，却没有让整个模型缓存成为常数。

### 混合模型的前缀快照为何比“保存 KV 页”更复杂？

MLA 层保留逐位置 latent/KV，KDA 层保留推进到某个前缀后的固定矩阵，短卷积还保留最近若干输入状态。一个有效 checkpoint 要让这些层表示同一个 token 边界。只有 MLA 页表，无法恢复已经被后续 token 覆盖的递归状态；只有最后状态，也无法回到任意较短前缀。

选择快照粒度影响容量与恢复：每个 token 都保存递归状态会失去固定状态容量优势，稀疏 checkpoint 需要恢复后重放部分 token。Prefix reuse、投机拒绝、抢占和 P/D 交接都应明确保存哪些状态、在哪个前缀边界，以及迟到写入怎样隔离。

资料：[DeepSeek-V3.2](https://arxiv.org/html/2512.02556v1)、[DeepSeek-V4 §2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)、[DeepSeek-V4 官方模型卡](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)、[Kimi Linear 官方说明](https://github.com/MoonshotAI/Kimi-Linear)、[Kimi Linear Technical Report](https://arxiv.org/html/2510.26692v1)。已有分析：[稀疏 Attention 的索引与执行](<../../../../../vllm/进阶篇/稀疏Attention：选择性访问与高效计算.md>)。

返回面试题：[06](../index.md#q11)
{.study-source}

</div>
