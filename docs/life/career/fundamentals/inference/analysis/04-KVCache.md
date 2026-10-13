---
title: "KV Cache：复用、容量与长上下文成本"
---

# KV Cache：复用、容量与长上下文成本

<div class="study-notes study-analysis" markdown="1">

来源：提供的「科大讯飞二面」「小米基模组一面」记录；对应推理优化[第 26～28 题](../index.md#q14)，补充 MHA 公式与 TP 分片。

## 1. 为什么历史 K/V 可以复用？ {#causal-reuse}

- <strong class="study-key">因果依赖。</strong> 固定模型与上下文下，标准因果 Attention 的历史 token 不依赖未来 token，其逐层 K/V 可保持不变。
- **新增输入。** 每轮只推进新增输入；新 Query 读取历史 KV，而不重新对整个历史执行投影和前向。已采样 token 在作为下一轮输入后才生成自身 KV。
- **仍需工作。** Dense Attention 的历史读取、QK 和加权求和仍然存在；缓存换掉的是重复历史前向，不是全部 Attention 成本。

对单个 head，在标准因果 Transformer 第 ℓ 层，位置 i 的隐藏状态只依赖位置 ≤i，因此固定前缀时它生成的 `kᶩ_i、vᶩ_i` 不会被未来位置 j>i 改写。新增位置 t 的 Attention 为 `softmax(q_t K_≤tᵀ/√d)V_≤t`：只新增当前位置的 Q/K/V，历史 K/V 可从缓存读取。

!!! tip "KV Cache 消除重复历史前向"
    缓存省掉历史 token 的逐层投影、FFN 等重复工作；**当前 Query 对历史的打分与 Value 加权仍然需要执行**。标准 dense Decode 对历史长度 S 的 Attention 成本仍随 S 增长。

这个证明依赖因果和相同输入状态。双向前缀尚未完整计算、adapter 改变、多模态 embedding 改变或位置语义不一致时，只有 token ID 相同不足以复用。每层缓存的是该层产生的 K/V，不能用最后一层缓存替代其他层。

返回面试题：[26](../index.md#q14)
{.study-source}

## 2. 怎样估算容量？ {#kv-capacity}

- **普通模型。** 相同层配置下，每 token 载荷约为 `2 × L × H_kv × d × s`，L 是层数，H_kv 是 KV heads，d 是 head dim，s 是元素字节数。
- **算例。** 32 层、8 KV heads、d=128、BF16，每 token 为 `2×32×8×128×2=131072 bytes`，即 128 KiB；4096 个缓存 token 为 512 MiB。
- **多请求与分片。** 乘以各请求缓存长度之和；前缀共享、TP 分片或复制改变物理占用，还要计页尾、对齐和临时空间。
- **模型差异。** GQA/MQA 用实际 KV heads；MLA、滑动窗口、混合层及不同 K/V 维度应按真实布局另算。

若 B 个请求均缓存 S 个 token，L 层都采用 MHA、H 个 Q/KV heads、相同 head_dim=d，则主载荷为：

$$
\mathrm{KVBytes}=2BSHd\,sL.
$$

这里 s 为单元素字节数，S 是实际缓存长度，不是只统计当前 chunk。MHA 普通 head TP=p 且 H 可等分时，每卡主载荷为总量/p。例：B=2、S=4096、H=32、d=128、L=32、BF16，每组逻辑载荷 2 GiB；TP=8 且无复制时每卡 256 MiB。

GQA 用 KV heads 代替 H，KV heads 少于 TP 时可能复制；MLA 不用此公式直接替换 Query heads。前缀共享、窗口层、量化 scale、页尾与对齐另计，见 [TP 的缓存布局](16-DPAttention与TBO.md)。

若 block size 为 b，每请求实际长度 S 需要 `ceil(S/b)` 个物理 block；未共享时物理主载荷按 `b·ceil(S/b)` 个 token 算。例：b=16，长度 17 的请求分配 32 个 token 容量，逻辑只有 17；长度 4096 时则没有这个尾部浪费。多请求总量应逐请求取整，再考虑共享页、保留页和量化元数据。

分页的价值是避免要求整条序列位于连续的大缓冲，并通过 block table 映射到离散物理页。它不压缩每个真实 token 的 K/V；Prefix reuse 则允许多个请求引用已完成的同一前缀页，需要引用计数和写入隔离。这两种机制解决的问题不同。

### 一个 token 怎样通过 block table 找到物理 KV？

令每块容纳 b 个 token，逻辑位置 t 的块号为 `j=floor(t/b)`、块内偏移为 `o=t mod b`。请求的 block table 在 j 处保存物理页 ID p，Attention kernel 再按本层、p、KV head、o 和维度计算地址。物理张量可采用不同排列，但逻辑到物理的两步映射必须一致。

例如 b=4，请求 table 为 `[7,2]`，第 5 个位置若从 0 开始记为 t=4，则落在物理页 2 的偏移 0；前四个 token 落在页 7。页 7 与页 2 无需连续。Kernel 不能把逻辑块 j=1 直接当物理页 1，取消后 table 也不能继续引用已复用页。

### 共享前缀后，新请求追加 token 会不会改坏别人的历史？

已完成的共享页必须保持相同历史语义。若两个请求共享一个只写入两项的末尾页，再在第三项分别追加不同 token，直接共写会冲突。可只共享已完整封闭的块，也可在追加前做 copy-on-write；具体策略按实现选择。引用计数保护页不被回收，COW 保护已有内容不被错误修改，两者不是同一个机制。

**分页解决地址和分配粒度，引用解决生命周期，写入隔离解决共享前缀后的分叉。**
{.study-core}

论文中的分页/COW 机制见[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)。不同模型有多组状态或不同 block 大小时，需要按各状态组分别映射，不能只维护一张普通 KV table。

返回面试题：[27](../index.md#q15)
{.study-source}

## 3. 长上下文的两种瓶颈 {#long-context}

- **容量。** 每请求缓存增大压缩并发空间，可能触发抢占、重算或卸载；分页减少分配浪费，不自动减少有效 KV 载荷。
- **执行。** Dense Decode 对更长历史进行访问和计算；带宽、Attention 并行组织及上下文长度共同决定时间。
- **方案匹配。** KV 量化降低字节数，但有误差和解码成本；稀疏或窗口访问减少参与历史，需符合模型语义；卸载扩大可用容量但引入传输。

沿用每 token 128 KiB 的 GQA 模型，单请求 4096 token 载荷为 512 MiB。若一轮 Decode 的后端确实读取全部这些 K/V，理论读取时间至少受 `512 MiB/BW` 限制；实际会受 head 复用、cache、量化、分片和 kernel 组织影响。多个长请求同时生成，会共同增加每轮历史访问与容量。

容量压力会进一步改变调度：更多请求入场可能提高权重复用，但使 KV 不足、触发抢占；恢复/重算再增加等待。性能曲线可能在容量阈值附近突然变差，不能把 batch 增加视为平滑而单调的吞吐提升。

### KV 不够时，抢占、重算与 offload 各保留什么？

重算式抢占释放物理状态但保留请求 token 历史，恢复时重新执行所需前向；offload 保留真实状态到 Host/外部介质，恢复时传回并重建映射。二者都要停止对旧页的新调度，并等在途读写结束后回收。重算消耗 GPU，offload 消耗带宽与容量，不能一律选择“缓存命中更多”的方案。

活动请求暂停还改变其他请求的 batch 与权重复用，恢复本身可能干扰 Decode。准入需留出输出增长空间和临时 workspace，不仅保证 prompt 当前能放下。对窗口 Attention，可按模型可见范围回收已不再使用的历史；对普通全局 Attention 则不能随意丢旧 KV。

相关笔记：[Attention 与历史 KV](../../../../../vllm/基础篇/Attention的计算机理与vLLM实现.md)。

返回面试题：[28](../index.md#q16)
{.study-source}

</div>
