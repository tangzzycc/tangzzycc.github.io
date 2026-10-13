---
title: "模型结构：面试问题与示例回答"
study_guide: true
---

# 模型结构

<div class="study-notes" markdown="1">

按“Attention 怎样保存历史→混合状态→MoE→多位置预测”复习，重点对应 Qwen3.5 混合状态接入、Qwen MoE 与投机解码工作。

13 道题 · 5 个主题 · 答案展开 · 每题附原理分析。
{.study-meta}

先掌握前四组的结构与状态差异；DSA、CSA/HCA 等具体模型变体作为进阶补充。
{.study-intro}

## 一、Attention 与缓存表示 {#attention-state}

### 01. MHA、GQA、MQA 有什么区别，为什么 GQA 能减少 KV Cache？ {#q1 .study-question}

MHA 每个 Query head 有对应的独立 KV head；GQA 让 <strong class="study-key">一组 Query heads 共享一组 K/V</strong>；MQA 则让所有 Query heads 共享一组 K/V。在层数、长度、head_dim 和 dtype 相同时，KV 载荷主要随 **KV head 数** 变化，因此 GQA/MQA 可减少缓存和历史读取。

Query head 数仍可保持不变，模型质量与实际带宽收益要结合训练和 kernel 实现判断。

[原理分析：从 KV Head 数比较](analysis/01-MHA-GQA与MLA.md#kv-heads)
{.study-analysis-link}

### 02. GQA/MLA、稀疏 Attention、序列压缩和线性 Attention 分别优化什么？ {#q7 .study-question}

GQA/MLA 主要减少 <strong class="study-key">每个历史位置的缓存表示</strong>；稀疏 Attention 减少 Query 访问的位置；序列压缩把多个位置表示为较少的缓存条目；递归式线性 Attention 用 **有限状态累计历史**，避免逐 token 保存完整 KV。它们作用的维度不同，可以组合，也各有质量和实现代价。

减少访问量不一定减少存储量，不能仅凭“高效 Attention”推断缓存复杂度。

[原理分析：先判断改变了哪个维度](analysis/03-稀疏压缩与线性Attention.md#compression-axis)
{.study-analysis-link}

### 03. MLA 是什么，与 GQA 的缓存压缩方式有什么区别？ {#q2 .study-question}

GQA 通过减少独立 KV heads 节省缓存；MLA 则将内容 K/V 联合表示为较小的 **共享 latent**，各 head 用自己的上投影生成不同内容 K/V。在 DeepSeek-V2/V3 一类设计中，执行侧可缓存 latent 与独立位置 Key，而非所有展开后的 K/V。

关键是 **低秩表示、矩阵吸收和位置分支配合**，它是模型设计，不能直接当作任意 MHA 模型的无损压缩开关。

[原理分析：MLA 共享 Latent，保留不同 Head 的投影](analysis/01-MHA-GQA与MLA.md#shared-latent)
{.study-analysis-link}

### 04. MLA 的矩阵吸收与解耦 RoPE 分别解决什么问题？ {#q3 .study-question}

只缓存 latent 后，如果每步仍展开全部历史 K/V，会新增大量计算。矩阵吸收把 **Key 上投影移到当前 Query，把 Value 上投影移到加权结果**，可直接用 latent 做 Attention。RoPE 的位置相关旋转会妨碍直接吸收，所以 DeepSeek 的 MLA 将位置分支单独处理。

核心是 **减少缓存同时避免重复展开历史**；不同执行后端可能采用不同计算路径。

[原理分析：为什么矩阵吸收能避免历史展开？](analysis/01-MHA-GQA与MLA.md#matrix-absorption) · [解耦 RoPE](analysis/01-MHA-GQA与MLA.md#decoupled-rope)
{.study-analysis-link}

继续深入：[MHA、GQA 与 MLA](analysis/01-MHA-GQA与MLA.md) · [稀疏、压缩与线性 Attention](analysis/03-稀疏压缩与线性Attention.md)
{.study-source}

## 二、混合与线性 Attention {#hybrid-attention}

### 05. Kimi Delta Attention（KDA）是什么，为什么适合长上下文 Decode？ {#q10 .study-question}

KDA 是 Kimi Linear 中的 **门控 Delta Rule 线性 Attention**，用通道级遗忘门控制递归状态，并通过纠错式更新写入新的 key/value 关联。该层每步读取和更新固定形状的历史状态，无需扫描全部逐 token KV，因此长上下文下状态成本更可控。

代价是 **有限状态容量与不同的模型语义**，它不等价于 softmax Attention，也不是所有 Kimi 模型都采用的结构。

[原理分析：KDA：用纠错式递归状态累计历史](analysis/03-稀疏压缩与线性Attention.md#delta-state)
{.study-analysis-link}

### 06. 为什么 Kimi Linear 混合 KDA 和 MLA，缓存是否变成常数？ {#q11 .study-question}

KDA 以较小递归状态承载历史，但有限状态对精细检索有约束；MLA 的全局 Attention 保留逐位置访问能力，因此混合设计兼顾 **状态效率与全局信息访问**。发布的 Kimi Linear 采用 3 层 KDA 对 1 层 MLA 的比例，这是模型配置。

KDA 层状态不随历史长度逐 token 增长，但 **MLA 层仍有随长度增长的缓存**，不能说整个模型缓存是常数。

[原理分析：混合模型仍有多种缓存](analysis/03-稀疏压缩与线性Attention.md#hybrid-state)
{.study-analysis-link}

继续深入：[稀疏、压缩与线性 Attention](analysis/03-稀疏压缩与线性Attention.md)
{.study-source}

## 三、MoE 与专家路由 {#moe}

### 07. MoE 的总参数量与激活参数量分别意味着什么？ {#q4 .study-question}

MoE 用 Router 为每个 token 选择部分专家，因此 **总参数量代表模型全部权重，激活参数量代表该 token 实际使用的参数范围**。激活量较小可降低部分 FFN 计算，但常驻部署通常仍需保存全部专家，不能只按激活参数量估算权重显存。

多个 token 还可能覆盖很多专家，所以运行速度取决于 **批次路由、权重读取、专家 GEMM 和通信**。

[原理分析：参数容量与实际工作不同](analysis/02-MoE与路由.md#total-active)
{.study-analysis-link}

### 08. MoE 的 Router、Top-k 和 Shared Expert 分别起什么作用？ {#q5 .study-question}

Router 根据 token 表示计算专家分数，选择 <strong class="study-key">top-k 路由专家</strong> 并按模型定义的权重合并输出；k 决定每 token 使用多少路由专家。Shared Expert 在采用该结构的模型中对所有 token 执行，提供公共能力，路由专家承担差异化处理。

它们通常替换 Transformer 中的 FFN，**并不让 Attention 或其他层也自动变稀疏**；分数归一化和共享分支组合取决于具体模型。

[原理分析：Router 决定模型使用哪些专家](analysis/02-MoE与路由.md#router-topk)
{.study-analysis-link}

### 09. 为什么 MoE 训练需要负载均衡，与推理时均衡有什么区别？ {#q6 .study-question}

Router 若集中选择少数专家，会导致 **专家训练不足、热点拥堵和设备利用率不均**。训练可使用均衡辅助损失，或模型规定的路由 bias 调节等机制，兼顾学习质量与专家使用。Serving 中还要处理短时热度和专家到设备的映射，常用重新放置与热点副本。

**逻辑专家的选择与物理专家的放置是两层问题**，推理均衡不能任意改变已训练模型的语义。

[原理分析：为什么要训练均衡？](analysis/02-MoE与路由.md#training-balance)
{.study-analysis-link}

继续深入：[MoE](analysis/02-MoE与路由.md)
{.study-source}

## 四、MTP 与候选依赖 {#mtp}

### 10. MTP（Multi-Token Prediction）推理头的基本思路是什么？ {#q12 .study-question}

MTP 让模型在下一 token 之外学习预测 <strong class="study-key">更多未来位置</strong>，可增加训练信号，并在支持的执行路径上用辅助模块产生投机候选。具体设计可用独立预测头，也可像 DeepSeek-V3 一样顺序连接模块、保留候选的因果链。

训练目标与推理使用是两层问题，**MTP 输出不等于已验证的最终 token**；是否加速仍取决于候选成本、目标验证和接受长度。

[原理分析：多位置训练目标与推理用途](analysis/04-MTP与半自回归.md#future-objective)
{.study-analysis-link}

### 11. 半自回归候选预测为什么可能比独立并行多头更准确？ {#q13 .study-question}

独立多头从同一状态直接预测多个偏移，可能缺少未来候选之间的条件信息；半自回归可让 **块内或组间候选依赖部分较早候选或状态**，更接近目标模型的因果条件，从而改善连续接受前缀。代价是增加部分串行计算，并可能传播候选误差。

“半自回归”不是唯一架构，应先说明 **哪些位置并行、哪些位置有依赖**，准确率与最终加速都需实测。

[原理分析：独立并行与部分自回归的取舍](analysis/04-MTP与半自回归.md#partial-autoregression)
{.study-analysis-link}

继续深入：[MTP 与半自回归](analysis/04-MTP与半自回归.md)
{.study-source}

## 五、稀疏 Attention 进阶 {#sparse-attention}

### 12. DeepSeek Sparse Attention（DSA）如何减少长上下文计算？ {#q8 .study-question}

DeepSeek-V3.2 的 DSA 用 **Lightning Indexer 筛选历史位置**，再让主 Attention 访问选中的 top-k KV，降低主 Attention 的读取与计算。Indexer 自身也有评分、缓存和选择成本；保留完整候选 KV 时，缓存仍可能随长度增长。

它按训练设计选择历史，**并非与完整 dense Attention 保证输出相同的执行优化**。

[原理分析：DSA：先索引，再访问主 KV](analysis/03-稀疏压缩与线性Attention.md#sparse-indexer)
{.study-analysis-link}

### 13. DeepSeek V4 的 CSA、HCA 与 DSA 有什么关系？ {#q9 .study-question}

CSA 是 **Compressed Sparse Attention**，先沿序列压缩 KV 条目，再用 DSA 选择部分压缩条目；HCA 是 **Heavily Compressed Attention**，采用更强序列压缩，对压缩条目做 dense Attention。两者还结合近期滑窗分支保留局部信息。

DSA 是选择访问位置的策略，CSA/HCA 是 V4 的混合结构，不能将三者当成同义词，也不能混同于 MLA 的 latent 特征压缩。

[原理分析：CSA/HCA：压缩的是序列条目](analysis/03-稀疏压缩与线性Attention.md#sequence-compression)
{.study-analysis-link}

继续深入：[稀疏、压缩与线性 Attention](analysis/03-稀疏压缩与线性Attention.md)
{.study-source}

</div>
